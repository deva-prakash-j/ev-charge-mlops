"""Baseline training with temporal splits and MLflow tracking.

Three models are trained in order of increasing complexity -- mean, ridge,
LightGBM -- so the gain from the candidate is measured against a floor rather
than asserted. Every run logs parameters, metrics, the git commit and the input
data hash, so any prediction is traceable to the code and data that produced it.

Usage:  python -m evcharge.models.train --source sample
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evcharge.config import ConfigError, Settings, get_settings, load_config
from evcharge.features.build import build_xy, fit as fit_features
from evcharge.models.evaluate import evaluate_slices, format_metrics, regression_metrics
from evcharge.validation.contract import apply_quarantine

logger = logging.getLogger(__name__)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def frame_hash(frame: pd.DataFrame) -> str:
    """Content hash of the training input, so a run names its exact dataset."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()


def load_sessions(settings: Settings, source: str) -> pd.DataFrame:
    files = glob.glob(
        str(settings.interim_dir / source / "sessions" / "**" / "*.parquet"),
        recursive=True,
    )
    if not files:
        raise ConfigError(
            f"No normalised sessions for {source!r}. Run: "
            f"python -m evcharge.ingest.normalise --source {source}"
        )
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frame.sort_values("connectionTimeLocal").reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame, split: dict[str, str]
) -> dict[str, pd.DataFrame]:
    when = pd.to_datetime(frame["connectionTimeLocal"])
    train_end = pd.Timestamp(split["train_end"])
    validation_end = pd.Timestamp(split["validation_end"])
    post_start = pd.Timestamp(split["post_shift_start"])
    post_end = pd.Timestamp(split["post_shift_end"])

    # Indices are reset because the native LightGBM library requires contiguous
    # label arrays, and sliced frames otherwise carry a gapped index.
    return {
        "train": frame[when <= train_end].reset_index(drop=True),
        "validation": frame[
            (when > train_end) & (when <= validation_end)
        ].reset_index(drop=True),
        "post_shift": frame[
            (when >= post_start) & (when <= post_end)
        ].reset_index(drop=True),
    }


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_mean": make_pipeline(
            SimpleImputer(strategy="median"), DummyRegressor(strategy="mean")
        ),
        "baseline_ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(**config["baselines"]["ridge"]),
        ),
        # Handles NaN natively, so no imputation is applied and the ~27% of
        # sessions without a driver request keep their missingness as signal.
        config["model"]["name"]: HistGradientBoostingRegressor(
            **config["model"]["params"]
        ),
    }


def run(settings: Settings, source: str) -> dict[str, Any]:
    import mlflow

    data_config = load_config("data")
    feature_config = load_config("features")
    model_config = load_config("model")
    contract_spec = load_config("contract")

    frame = load_sessions(settings, source)
    frame, quarantined = apply_quarantine(frame, contract_spec["quarantine_rules"])
    frame = frame[frame[feature_config["target"]].notna()].reset_index(drop=True)
    logger.info("%d sessions after quarantine (%d removed)", len(frame), len(quarantined))

    splits = temporal_split(frame, model_config["split"])
    for name, part in splits.items():
        logger.info("%-11s %6d rows", name, len(part))
    if splits["train"].empty or splits["validation"].empty:
        raise ConfigError("Temporal split produced an empty train or validation set.")

    # Fitted on the training slice only; fitting on all data would leak the
    # future category vocabulary into training.
    spec = fit_features(splits["train"], feature_config, data_config)

    prepared = {
        name: build_xy(part, spec, feature_config, data_config)
        for name, part in splits.items()
        if not part.empty
    }
    x_train, y_train = prepared["train"]
    y_train = np.ascontiguousarray(y_train.to_numpy(dtype="float64"))

    mlflow.set_tracking_uri((settings.artifacts_dir.parent / model_config["mlflow"]["tracking_uri"]).as_uri())
    mlflow.set_experiment(model_config["mlflow"]["experiment"])

    commit = git_commit()
    data_hash = frame_hash(frame)
    results: dict[str, Any] = {}

    for name, estimator in build_models(model_config).items():
        with mlflow.start_run(run_name=f"{name}-{source}"):
            estimator.fit(x_train, y_train)

            mlflow.log_params(
                {
                    "model": name,
                    "source": source,
                    "git_commit": commit,
                    "data_sha256": data_hash[:16],
                    "n_features": x_train.shape[1],
                    "n_train": len(x_train),
                    "train_end": model_config["split"]["train_end"],
                    "validation_end": model_config["split"]["validation_end"],
                    "feature_spec_version": spec.version,
                }
            )
            if name == model_config["model"]["name"]:
                mlflow.log_params(
                    {f"hgb_{k}": v for k, v in model_config["model"]["params"].items()}
                )

            run_metrics: dict[str, Any] = {}
            for split_name, (features, target) in prepared.items():
                predictions = estimator.predict(features)
                sliced = evaluate_slices(
                    splits[split_name],
                    target,
                    np.asarray(predictions),
                    model_config["evaluation"]["breakdowns"],
                )
                run_metrics[split_name] = sliced
                for metric, value in sliced["overall"].items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        mlflow.log_metric(f"{split_name}_{metric}", float(value))

            spec_path = settings.artifacts_dir / f"feature_spec_{source}.json"
            spec.save(spec_path)
            mlflow.log_artifact(str(spec_path))

            if name == model_config["model"]["name"]:
                # Permutation importance on validation, not training: it measures
                # what the model actually relies on to generalise.
                x_validation, y_validation = prepared["validation"]
                importance = permutation_importance(
                    estimator,
                    x_validation,
                    y_validation,
                    n_repeats=5,
                    random_state=42,
                    scoring="neg_mean_absolute_error",
                )
                run_metrics["permutation_importance"] = (
                    pd.Series(importance.importances_mean, index=x_train.columns)
                    .sort_values(ascending=False)
                    .round(4)
                    .to_dict()
                )
                # Signature and example are logged so the V2 serving layer can
                # validate request payloads against the trained schema.
                example = x_train.head(5)
                signature = mlflow.models.infer_signature(
                    example, estimator.predict(example)
                )
                mlflow.sklearn.log_model(
                    estimator,
                    artifact_path="model",
                    signature=signature,
                    input_example=example,
                )

            results[name] = run_metrics

    report = {
        "source": source,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "data_sha256": data_hash,
        "rows": {k: len(v) for k, v in splits.items()},
        "feature_names": spec.feature_names,
        "results": results,
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / f"training_{source}.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--source", choices=("sample", "raw"), default="sample")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for noisy in ("mlflow", "py.warnings"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    try:
        settings = get_settings()
        report = run(settings, args.source)
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(f"\n=== results ({args.source}) ===")
    for model, splits in report["results"].items():
        print(f"\n{model}")
        for split_name in ("train", "validation", "post_shift"):
            if split_name in splits:
                print("  " + format_metrics(split_name, splits[split_name]["overall"]))

    best = min(
        report["results"],
        key=lambda m: report["results"][m]["validation"]["overall"]["mae"],
    )
    print(f"\n[ok]   best on validation MAE: {best}")
    print(f"[ok]   report -> {settings.artifacts_dir / f'training_{args.source}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
