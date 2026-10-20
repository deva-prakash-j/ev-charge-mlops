"""Local MLflow Model Registry aliases and promotion quality gates."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyRegressor

from evcharge.config import SOURCES, ConfigError, Settings, get_settings, load_config
from evcharge.features.build import FeatureSpec, build_xy, fit as fit_features, transform
from evcharge.models.evaluate import regression_metrics

logger = logging.getLogger(__name__)


class RegistryError(RuntimeError):
    """Raised when the registry cannot safely complete a promotion operation."""


def resolve_tracking_uri(
    settings: Settings | None = None,
    model_config: dict[str, Any] | None = None,
) -> str:
    """Use an explicit deployment URI, otherwise the repository-local store."""
    configured = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured

    settings = settings or get_settings()
    model_config = model_config or load_config("model")
    configured = str(model_config["mlflow"]["tracking_uri"])
    if "://" in configured:
        return configured

    tracking_path = Path(configured)
    if not tracking_path.is_absolute():
        tracking_path = settings.artifacts_dir.parent / tracking_path
    return tracking_path.resolve().as_uri()


def registered_model_name(source: str, target: str, prefix: str = "evcharge") -> str:
    if source not in SOURCES or target not in {"energy", "dwell"}:
        raise ConfigError(
            f"Unsupported registry source/target pair: {source!r}/{target!r}."
        )
    return f"{prefix}-{source}-{target}"


def evaluate_promotion_gate(
    candidate_mae: float,
    mean_baseline_mae: float,
    ridge_baseline_mae: float,
    production_mae: float | None,
    max_production_regression: float = 0.10,
) -> dict[str, Any]:
    """Require a candidate to beat both floors and cap regression vs Production."""
    values = {
        "candidate_mae": candidate_mae,
        "mean_baseline_mae": mean_baseline_mae,
        "ridge_baseline_mae": ridge_baseline_mae,
    }
    if production_mae is not None:
        values["production_mae"] = production_mae
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ConfigError("Promotion MAEs must be finite, non-negative numbers.")
    if (
        not math.isfinite(max_production_regression)
        or max_production_regression < 0
    ):
        raise ConfigError("Maximum Production regression must be finite and non-negative.")

    checks = {
        "beats_mean_baseline": {
            "passed": candidate_mae < mean_baseline_mae,
            "candidate_mae": candidate_mae,
            "baseline_mae": mean_baseline_mae,
        },
        "beats_ridge_baseline": {
            "passed": candidate_mae < ridge_baseline_mae,
            "candidate_mae": candidate_mae,
            "baseline_mae": ridge_baseline_mae,
        },
    }
    if production_mae is None:
        checks["production_regression_within_limit"] = {
            "passed": True,
            "production_mae": None,
            "max_allowed_mae": None,
            "bootstrap": True,
        }
    else:
        max_allowed_mae = production_mae * (1 + max_production_regression)
        checks["production_regression_within_limit"] = {
            "passed": candidate_mae <= max_allowed_mae,
            "production_mae": production_mae,
            "max_allowed_mae": max_allowed_mae,
            "max_regression_fraction": max_production_regression,
            "bootstrap": False,
        }

    failed = [name for name, check in checks.items() if not check["passed"]]
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def current_production_evaluation(
    client: MlflowClient,
    model_name: str,
    validation_frame: pd.DataFrame,
    y_validation: pd.Series,
    feature_config: dict[str, Any],
    data_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Score Production on this candidate's held-out rows using its fitted spec."""
    try:
        registered = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise

    production_version = registered.aliases.get("production")
    if production_version is None:
        return None

    version = client.get_model_version(model_name, str(production_version))
    feature_spec_artifact = version.tags.get("feature_spec_artifact")
    if not feature_spec_artifact:
        raise RegistryError(
            f"Production version {model_name}/{version.version} has no "
            "feature_spec_artifact tag; refusing an incomparable quality check."
        )

    spec_path = Path(client.download_artifacts(version.run_id, feature_spec_artifact))
    production_spec = FeatureSpec.load(spec_path)
    mlflow.set_tracking_uri(client.tracking_uri)
    production_model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
    production_features = transform(
        validation_frame, production_spec, feature_config, data_config
    )
    score = regression_metrics(
        y_validation, np.asarray(production_model.predict(production_features))
    )
    if not math.isfinite(score["mae"]):
        raise RegistryError(
            f"Production model {model_name}/{version.version} produced an invalid "
            "validation MAE."
        )
    return {
        "version": str(version.version),
        "run_id": version.run_id,
        "validation_mae": score["mae"],
        "rows": score["n"],
    }


def _append_decision(path: Path, decision: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(decision, sort_keys=True, allow_nan=False) + "\n")


def register_candidate(
    client: MlflowClient,
    *,
    model_name: str,
    source: str,
    target: str,
    run_id: str,
    feature_spec_artifact: str,
    candidate_mae: float,
    mean_baseline_mae: float,
    ridge_baseline_mae: float,
    production: dict[str, Any] | None,
    gate: dict[str, Any],
    decision_log: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Register every candidate, stage it, and promote only if its gate passes."""
    try:
        client.create_registered_model(
            model_name,
            tags={"project": "evcharge", "source": source, "target": target},
            description=f"EV charge-session {target} predictor for {source} data.",
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        tags={
            "source": source,
            "target": target,
            "feature_spec_artifact": feature_spec_artifact,
            "candidate_validation_mae": str(candidate_mae),
            "mean_baseline_validation_mae": str(mean_baseline_mae),
            "ridge_baseline_validation_mae": str(ridge_baseline_mae),
            "production_validation_mae": (
                "" if production is None else str(production["validation_mae"])
            ),
        },
        description=(
            f"{target} candidate; validation MAE={candidate_mae:.6f}; "
            f"source={source}; git commit={evidence['git_commit']}."
        ),
        await_creation_for=300,
    )
    version_number = str(version.version)
    client.set_registered_model_alias(model_name, "staging", version_number)

    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "source": source,
        "target": target,
        "candidate_version": version_number,
        "candidate_run_id": run_id,
        "candidate_validation_mae": candidate_mae,
        "mean_baseline_validation_mae": mean_baseline_mae,
        "ridge_baseline_validation_mae": ridge_baseline_mae,
        "previous_production": production,
        "gate": gate,
        **evidence,
    }

    client.set_model_version_tag(
        model_name, version_number, "promotion_gate", "passed" if gate["passed"] else "blocked"
    )
    if not gate["passed"]:
        decision["status"] = "blocked"
        client.set_model_version_tag(model_name, version_number, "promotion_status", "blocked")
        _append_decision(decision_log, decision)
        logger.info(
            "Candidate %s version %s blocked: %s",
            model_name,
            version_number,
            ", ".join(gate["failed_checks"]),
        )
        return decision

    decision["status"] = "gate_passed"
    _append_decision(decision_log, decision)
    try:
        client.set_registered_model_alias(model_name, "production", version_number)
    except MlflowException as exc:
        failure = {
            **decision,
            "status": "promotion_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _append_decision(decision_log, failure)
        raise RegistryError(
            f"Quality gate passed but Production alias update failed for "
            f"{model_name}/{version_number}: {exc}"
        ) from exc

    client.set_model_version_tag(
        model_name, version_number, "promotion_status", "promoted"
    )
    promoted = {**decision, "status": "promoted"}
    _append_decision(decision_log, promoted)
    logger.info("Promoted %s version %s to Production.", model_name, version_number)
    return promoted


def feature_spec_artifact(source: str, target: str) -> str:
    return (
        f"feature_spec_{source}.json"
        if target == "energy"
        else f"feature_spec_{target}_{source}.json"
    )


def prove_degraded_candidate(
    settings: Settings, source: str, target: str
) -> dict[str, Any]:
    """Log a constant-zero model and prove the promotion gate blocks it."""
    from evcharge.models import train
    from evcharge.validation.contract import apply_quarantine

    if source not in SOURCES or target not in {"energy", "dwell"}:
        raise ConfigError(
            f"Proof requires a source in {SOURCES} and target energy/dwell."
        )
    report_path = settings.artifacts_dir / (
        f"training_{source}.json"
        if target == "energy"
        else f"training_{target}_{source}.json"
    )
    if not report_path.is_file():
        raise ConfigError(
            f"Training report not found: {report_path}. Run the pipeline first."
        )
    training_report = json.loads(report_path.read_text(encoding="utf-8"))

    data_config = load_config("data")
    feature_config = load_config("features")
    model_config = load_config("model")
    contract_config = load_config("contract")
    target_column = (
        feature_config["target"]
        if target == "energy"
        else feature_config["dwell_target"]
    )
    frame = train.load_sessions(settings, source)
    frame, _ = apply_quarantine(frame, contract_config["quarantine_rules"])
    frame = frame[frame[target_column].notna()].reset_index(drop=True)
    if training_report.get("source") != source or training_report.get("target") != target_column:
        raise ConfigError(
            f"Training report {report_path} does not match source {source!r} "
            f"and target {target_column!r}."
        )
    data_hash = train.frame_hash(frame)
    if training_report.get("data_sha256") != data_hash:
        raise ConfigError(
            f"Training report {report_path} has a different data fingerprint; "
            "retrain before running the gate proof."
        )
    splits = train.temporal_split(frame, model_config["split"])
    if splits["train"].empty or splits["validation"].empty:
        raise ConfigError("Gate proof requires non-empty train and validation splits.")
    actual_rows = {name: len(part) for name, part in splits.items()}
    if training_report.get("rows") != actual_rows:
        raise ConfigError(
            f"Training report {report_path} has split counts that no longer match "
            "the configured temporal boundaries; retrain before running the proof."
        )
    feature_spec = fit_features(splits["train"], feature_config, data_config)
    x_train, y_train = build_xy(
        splits["train"], feature_spec, feature_config, data_config, target_column
    )
    x_validation, y_validation = build_xy(
        splits["validation"], feature_spec, feature_config, data_config, target_column
    )

    degraded = DummyRegressor(strategy="constant", constant=0.0).fit(x_train, y_train)
    candidate_mae = regression_metrics(
        y_validation, np.asarray(degraded.predict(x_validation))
    )["mae"]
    tracking_uri = resolve_tracking_uri(settings, model_config)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = model_config["mlflow"]["experiment"]
    if target == "dwell":
        experiment = f"{experiment}-dwell"
    mlflow.set_experiment(experiment)

    spec_artifact = f"feature_spec_{source}_{target}_degraded.json"
    spec_path = settings.artifacts_dir / spec_artifact
    feature_spec.save(spec_path)
    example = x_train.head(5)
    signature = mlflow.models.infer_signature(example, degraded.predict(example))
    with mlflow.start_run(run_name=f"degraded-proof-{source}-{target}") as run:
        mlflow.log_params(
            {
                "model": "deliberately_degraded_constant_zero",
                "source": source,
                "target": target_column,
                "proof_candidate": True,
                "git_commit": training_report["git_commit"],
            }
        )
        mlflow.log_metric("validation_mae", candidate_mae)
        mlflow.log_artifact(str(spec_path))
        mlflow.sklearn.log_model(
            degraded,
            artifact_path="model",
            signature=signature,
            input_example=example,
        )
        candidate_run_id = run.info.run_id

    model_name = registered_model_name(
        source, target, model_config["registry"]["registered_model_prefix"]
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    production = current_production_evaluation(
        client,
        model_name,
        splits["validation"],
        y_validation,
        feature_config,
        data_config,
    )
    metrics = training_report["results"]
    mean_mae = metrics["baseline_mean"]["validation"]["overall"]["mae"]
    ridge_mae = metrics["baseline_ridge"]["validation"]["overall"]["mae"]
    gate = evaluate_promotion_gate(
        candidate_mae,
        mean_mae,
        ridge_mae,
        None if production is None else production["validation_mae"],
        model_config["registry"]["max_production_mae_regression"],
    )
    if gate["passed"]:
        raise RegistryError(
            "The deliberate constant-zero proof model unexpectedly passed the gate; "
            "it was not submitted to the registry."
        )

    decision = register_candidate(
        client,
        model_name=model_name,
        source=source,
        target=target,
        run_id=candidate_run_id,
        feature_spec_artifact=spec_artifact,
        candidate_mae=candidate_mae,
        mean_baseline_mae=mean_mae,
        ridge_baseline_mae=ridge_mae,
        production=production,
        gate=gate,
        decision_log=settings.artifacts_dir
        / model_config["registry"]["decision_log"],
        evidence={
            "git_commit": training_report["git_commit"],
            "data_sha256": data_hash,
            "proof_candidate": "deliberately_degraded_constant_zero",
            "validation_rows": len(splits["validation"]),
        },
    )
    if decision["status"] != "blocked":
        raise RegistryError(
            f"Expected the degraded proof candidate to be blocked, got "
            f"{decision['status']!r}."
        )
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove the model promotion gate.")
    parser.add_argument("--source", choices=SOURCES, default="raw")
    parser.add_argument("--target", choices=("energy", "dwell"), default="energy")
    parser.add_argument(
        "--prove-degraded-candidate",
        action="store_true",
        help="Register a constant-zero candidate in Staging and verify it is blocked.",
    )
    args = parser.parse_args(argv)
    if not args.prove_degraded_candidate:
        parser.error("pass --prove-degraded-candidate to run the local gate proof")
    try:
        decision = prove_degraded_candidate(get_settings(), args.source, args.target)
    except (ConfigError, RegistryError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(
        f"[ok]   deliberately degraded candidate blocked: "
        f"{decision['model_name']} version {decision['candidate_version']}"
    )
    print(
        f"[ok]   validation MAE {decision['candidate_validation_mae']:.4f}; "
        f"failed checks: {', '.join(decision['gate']['failed_checks'])}"
    )
    print("[ok]   evidence -> artifacts/promotion_decisions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
