"""Energy and dwell-time training with temporal splits and MLflow tracking.

Three models are trained in order of increasing complexity -- mean, ridge,
HistGradientBoosting -- so the gain from the candidate is measured against a
floor rather than asserted. Every run logs parameters, metrics, the git commit
and the input data hash, so any prediction is traceable to the code and data
that produced it.

Usage:  python -m evcharge.models.train --source sample [--target energy|dwell]
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
from pathlib import Path
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

    # Reset indices so each temporal slice has a clean, contiguous row index.
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


def persist_best_model(
    settings: Settings,
    source: str,
    target: str,
    name: str,
    estimator: Any,
    spec: Any,
    metrics: dict[str, Any],
) -> Path:
    """Write the selected model next to the exact feature spec it was fitted with.

    Saving them together is what makes the artefact loadable later: the model
    alone cannot reproduce its own input encoding.
    """
    import joblib

    model_dir = settings.models_dir / (
        source if target == "energy" else f"{source}_{target}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(estimator, model_dir / "model.joblib")
    spec.save(model_dir / "feature_spec.json")
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    target_description = (
        "kWhDelivered (energy delivered in a charging session)"
        if target == "energy"
        else "connectedHours (elapsed connection-to-disconnection hours)"
    )
    (model_dir / "MODEL_CARD.md").write_text(
        f"# Model card - {name} ({source}, {target})\n\n"
        f"- Target: {target_description}\n"
        f"- Trained: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"- Git commit: {git_commit()[:12]}\n"
        f"- Features: {len(spec.feature_names)}\n"
        f"- Validation MAE: {metrics['validation']['overall']['mae']:.4f}\n"
        f"- Post-shift MAE: {metrics.get('post_shift', {}).get('overall', {}).get('mae', float('nan')):.4f}\n\n"
        "Inputs are plug-in-time only. Post-hoc columns (disconnectTime, "
        "doneChargingTime) are rejected by the feature builder.\n\n"
        "Load with:\n\n"
        "```python\n"
        "import joblib\n"
        "from evcharge.features.build import FeatureSpec\n"
        f"model = joblib.load('models/{model_dir.name}/model.joblib')\n"
        f"spec = FeatureSpec.load('models/{model_dir.name}/feature_spec.json')\n"
        "```\n",
        encoding="utf-8",
    )
    return model_dir


def run(
    settings: Settings, source: str, target: str = "energy"
) -> dict[str, Any]:
    import mlflow

    if target not in {"energy", "dwell"}:
        raise ConfigError(f"Unsupported target {target!r}; choose energy or dwell.")

    data_config = load_config("data")
    feature_config = load_config("features")
    model_config = load_config("model")
    contract_spec = load_config("contract")
    target_column = (
        feature_config["target"]
        if target == "energy"
        else feature_config["dwell_target"]
    )

    frame = load_sessions(settings, source)
    frame, quarantined = apply_quarantine(frame, contract_spec["quarantine_rules"])
    frame = frame[frame[target_column].notna()].reset_index(drop=True)
    logger.info("%d sessions after quarantine (%d removed)", len(frame), len(quarantined))

    splits = temporal_split(frame, model_config["split"])
    for name, part in splits.items():
        logger.info("%-11s %6d rows", name, len(part))
    if splits["train"].empty or splits["validation"].empty:
        raise ConfigError("Temporal split produced an empty train or validation set.")
    if splits["post_shift"].empty:
        post_start = model_config["split"]["post_shift_start"]
        post_end = model_config["split"]["post_shift_end"]
        raise ConfigError(
            "Temporal split produced no labeled post-shift rows in "
            f"{post_start}..{post_end}; check the configured window and source data."
        )

    # Fitted on the training slice only; fitting on all data would leak the
    # future category vocabulary into training.
    spec = fit_features(splits["train"], feature_config, data_config)

    prepared = {
        name: build_xy(part, spec, feature_config, data_config, target=target_column)
        for name, part in splits.items()
        if not part.empty
    }
    x_train, y_train = prepared["train"]
    y_train = np.ascontiguousarray(y_train.to_numpy(dtype="float64"))

    from evcharge.models.registry import resolve_tracking_uri

    tracking_uri = resolve_tracking_uri(settings, model_config)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = model_config["mlflow"]["experiment"]
    if target == "dwell":
        experiment = f"{experiment}-dwell"
    mlflow.set_experiment(experiment)

    commit = git_commit()
    data_hash = frame_hash(frame)
    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    run_ids: dict[str, str] = {}

    for name, estimator in build_models(model_config).items():
        with mlflow.start_run(run_name=f"{name}-{source}") as run:
            estimator.fit(x_train, y_train)
            fitted[name] = estimator
            run_ids[name] = run.info.run_id

            mlflow.log_params(
                {
                    "model": name,
                    "source": source,
                    "target": target_column,
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
            for split_name, (features, target_values) in prepared.items():
                predictions = estimator.predict(features)
                sliced = evaluate_slices(
                    splits[split_name],
                    target_values,
                    np.asarray(predictions),
                    model_config["evaluation"]["breakdowns"],
                )
                run_metrics[split_name] = sliced
                for metric, value in sliced["overall"].items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        mlflow.log_metric(f"{split_name}_{metric}", float(value))

            spec_path = settings.artifacts_dir / (
                f"feature_spec_{source}.json"
                if target == "energy"
                else f"feature_spec_{target}_{source}.json"
            )
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
            # Log every fitted model so whichever candidate wins the same-
            # validation comparison can be registered without retraining.
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

    best = min(
        results, key=lambda m: results[m]["validation"]["overall"]["mae"]
    )
    model_dir = persist_best_model(
        settings, source, target, best, fitted[best], spec, results[best]
    )
    logger.info("best model %s -> %s", best, model_dir)

    report = {
        "source": source,
        "target": target_column,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "data_sha256": data_hash,
        "rows": {k: len(v) for k, v in splits.items()},
        "feature_names": spec.feature_names,
        "best_model": best,
        "model_dir": str(model_dir),
        "results": results,
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_name = (
        f"training_{source}.json"
        if target == "energy"
        else f"training_{target}_{source}.json"
    )
    report_path = settings.artifacts_dir / report_name
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    from evcharge.models.registry import (
        current_production_evaluation,
        evaluate_promotion_gate,
        feature_spec_artifact,
        register_candidate,
        registered_model_name,
    )
    from mlflow import MlflowClient

    registry_config = model_config["registry"]
    model_name = registered_model_name(
        source, target, registry_config["registered_model_prefix"]
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    production = current_production_evaluation(
        client,
        model_name,
        splits["validation"],
        prepared["validation"][1],
        feature_config,
        data_config,
    )
    mean_mae = results["baseline_mean"]["validation"]["overall"]["mae"]
    ridge_mae = results["baseline_ridge"]["validation"]["overall"]["mae"]
    candidate_mae = results[best]["validation"]["overall"]["mae"]
    gate = evaluate_promotion_gate(
        candidate_mae,
        mean_mae,
        ridge_mae,
        None if production is None else production["validation_mae"],
        registry_config["max_production_mae_regression"],
    )
    report["promotion"] = register_candidate(
        client,
        model_name=model_name,
        source=source,
        target=target,
        run_id=run_ids[best],
        feature_spec_artifact=feature_spec_artifact(source, target),
        candidate_mae=candidate_mae,
        mean_baseline_mae=mean_mae,
        ridge_baseline_mae=ridge_mae,
        production=production,
        gate=gate,
        decision_log=settings.artifacts_dir / registry_config["decision_log"],
        evidence={
            "git_commit": commit,
            "data_sha256": data_hash,
            "validation_rows": len(splits["validation"]),
            "candidate_model": best,
        },
    )
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--source", choices=("sample", "raw"), default="sample")
    parser.add_argument(
        "--target",
        choices=("energy", "dwell"),
        default="energy",
        help="Predict delivered energy or elapsed connected duration.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for noisy in ("mlflow", "py.warnings"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    try:
        settings = get_settings()
        report = run(settings, args.source, args.target)
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
    promotion = report["promotion"]
    if promotion["status"] == "promoted":
        print(
            f"[ok]   promoted {promotion['model_name']} version "
            f"{promotion['candidate_version']} to Production"
        )
    else:
        print(
            f"[warn] candidate {promotion['model_name']} version "
            f"{promotion['candidate_version']} remains in Staging; gate failed: "
            f"{', '.join(promotion['gate']['failed_checks'])}"
        )
    report_name = (
        f"training_{args.source}.json"
        if args.target == "energy"
        else f"training_{args.target}_{args.source}.json"
    )
    print(f"[ok]   report -> {settings.artifacts_dir / report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
