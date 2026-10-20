"""Evidently drift scoring for monthly EV charge-session batches."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from scipy.stats import ks_2samp

from evcharge.config import ConfigError, Settings, load_config
from evcharge.features.build import FeatureSpec, transform
from evcharge.models.registry import (
    registered_model_name,
    resolve_tracking_uri,
)

logger = logging.getLogger(__name__)

PSI_ALERT_THRESHOLD = 0.2
KS_ALERT_THRESHOLD = 0.01
CONTINUOUS_FEATURES = {
    "arrival_hour",
    "arrival_hour_sin",
    "arrival_hour_cos",
    "declared_duration_minutes",
    "declared_kwh_per_minute",
    "kWhRequested",
    "milesRequested",
    "minutesAvailable",
    "WhPerMile",
    "kWhDelivered",
    "predicted_kwh",
}
CATEGORICAL_FEATURES = {
    "arrival_dayofweek",
    "arrival_is_weekend",
    "arrival_month",
    "has_user_inputs",
    "siteID_code",
    "clusterID_code",
    "stationID_code",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(settings: Settings) -> pd.DataFrame:
    """Load the frozen snapshot only after verifying its data-free manifest."""
    reference_dir = settings.data_dir / "reference"
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigError(
            f"Drift reference manifest is missing: {manifest_path}. "
            "Use the licensed evaluation copy and regenerate the reference."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read drift reference manifest: {manifest_path}") from exc

    reference_path = reference_dir / manifest.get("file", "")
    if not reference_path.is_file():
        raise ConfigError(f"Drift reference snapshot is missing: {reference_path}")
    if _sha256(reference_path) != manifest.get("sha256"):
        raise ConfigError(f"Drift reference checksum mismatch: {reference_path}")

    frame = pd.read_parquet(reference_path)
    if len(frame) != manifest.get("rows") or frame.columns.tolist() != manifest.get(
        "columns"
    ):
        raise ConfigError(f"Drift reference does not match its manifest: {reference_path}")
    required = {
        "connectionTimeLocal",
        "siteID",
        "clusterID",
        "stationID",
        "declared_duration_minutes",
        "kWhDelivered",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ConfigError(
            f"Drift reference is missing required monitoring fields: {missing}. "
            "Regenerate it with the current validation.reference module."
        )
    return frame


def load_production_model(
    settings: Settings, source: str = "raw"
) -> tuple[Any, FeatureSpec, str, str]:
    """Load the model and feature spec from the source's Production alias."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    model_config = load_config("model")
    model_name = registered_model_name(
        source,
        "energy",
        model_config["registry"]["registered_model_prefix"],
    )
    tracking_uri = resolve_tracking_uri(settings, model_config)
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException as exc:
        raise ConfigError(
            f"No Production alias is registered for {model_name!r} at "
            f"{tracking_uri}. Run the contract-gated training flow first."
        ) from exc

    artifact_path = version.tags.get("feature_spec_artifact")
    if not artifact_path:
        raise ConfigError(
            f"Production model {model_name!r} version {version.version} "
            "has no feature_spec_artifact tag."
        )
    spec_path = Path(client.download_artifacts(version.run_id, artifact_path))
    model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
    return model, FeatureSpec.load(spec_path), model_name, str(version.version)


def _prediction_frame(
    frame: pd.DataFrame,
    model: Any,
    feature_spec: FeatureSpec,
    *,
    reference: bool,
) -> pd.DataFrame:
    feature_config = load_config("features")
    data_config = load_config("data")
    model_frame = frame.copy()
    if reference and "connectionTime" not in model_frame:
        connection = pd.to_datetime(model_frame["connectionTimeLocal"])
        model_frame["connectionTime"] = connection
        model_frame["requestedDeparture"] = connection + pd.to_timedelta(
            model_frame["declared_duration_minutes"], unit="m"
        )

    model_features = transform(
        model_frame, feature_spec, feature_config, data_config
    )
    predictions = model.predict(model_features)
    features = model_features.copy()
    for column in ("siteID", "clusterID", "stationID"):
        encoded = f"{column}_code"
        if encoded in features:
            values = frame[column].astype("string").fillna("unknown")
            features[encoded] = values.astype(str)
    for column in CATEGORICAL_FEATURES:
        if column in features:
            features[column] = features[column].map(
                lambda value: "unknown" if pd.isna(value) else str(value)
            )

    result = features.copy()
    result["kWhDelivered"] = pd.to_numeric(
        frame.get("kWhDelivered"), errors="coerce"
    ).to_numpy()
    result["predicted_kwh"] = np.asarray(predictions, dtype="float64")
    return result


def _score_metrics(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reference.empty or current.empty:
        raise ValueError("Drift scoring requires non-empty reference and current data.")
    missing_columns = sorted(set(reference.columns) ^ set(current.columns))
    if missing_columns:
        raise ValueError(
            f"Reference and current feature schemas differ: {missing_columns}"
        )

    continuous_columns = {
        column
        for column in reference.columns
        if column in CONTINUOUS_FEATURES
        or (
            column not in CATEGORICAL_FEATURES
            and pd.api.types.is_numeric_dtype(reference[column])
            and reference[column].nunique(dropna=True) > 2
        )
    }
    metrics = []
    for column in reference.columns:
        ref_values = reference[column].replace([np.inf, -np.inf], np.nan).dropna()
        cur_values = current[column].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref_values) < 2 or len(cur_values) < 2:
            continue
        methods = (
            ("psi", "ks", "jensenshannon")
            if column in continuous_columns
            else ("psi", "jensenshannon")
        )
        for method in methods:
            threshold = (
                PSI_ALERT_THRESHOLD
                if method == "psi"
                else KS_ALERT_THRESHOLD
                if method == "ks"
                else None
            )
            metrics.append(
                ValueDrift(column=column, method=method, threshold=threshold)
            )

    if not metrics:
        raise ValueError("No comparable columns have at least two non-missing values.")

    snapshot = Report(metrics=metrics).run(
        current_data=current, reference_data=reference
    )
    values = {
        (item["config"]["column"], item["config"]["method"]): float(item["value"])
        for item in snapshot.dict()["metrics"]
    }
    scores: list[dict[str, Any]] = []
    for column in reference.columns:
        ref_values = reference[column].replace([np.inf, -np.inf], np.nan).dropna()
        cur_values = current[column].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref_values) < 2 or len(cur_values) < 2:
            scores.append(
                {
                    "column": column,
                    "status": "insufficient_non_missing_values",
                    "reference_rows": len(ref_values),
                    "current_rows": len(cur_values),
                    "reference_missing_fraction": (
                        1 - len(ref_values) / len(reference)
                    ),
                    "current_missing_fraction": 1 - len(cur_values) / len(current),
                    "psi": None,
                    "ks_statistic": None,
                    "ks_pvalue": None,
                    "js_distance": None,
                    "psi_alert": False,
                    "ks_alert": False,
                }
            )
            continue
        psi = values[(column, "psi")]
        js_distance = values[(column, "jensenshannon")]
        if column in continuous_columns:
            ks_result = ks_2samp(
                ref_values.to_numpy(dtype="float64"),
                cur_values.to_numpy(dtype="float64"),
            )
            ks_statistic = float(ks_result.statistic)
            ks_pvalue = values[(column, "ks")]
        else:
            ks_statistic = None
            ks_pvalue = None

        scores.append(
            {
                "column": column,
                "status": "scored",
                "reference_rows": len(ref_values),
                "current_rows": len(cur_values),
                "reference_missing_fraction": 1 - len(ref_values) / len(reference),
                "current_missing_fraction": 1 - len(cur_values) / len(current),
                "psi": psi,
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "js_distance": js_distance,
                "psi_alert": psi > PSI_ALERT_THRESHOLD,
                "ks_alert": ks_pvalue is not None
                and ks_pvalue < KS_ALERT_THRESHOLD,
            }
        )

    score_by_column = {score["column"]: score for score in scores}
    target = score_by_column.get("kWhDelivered", {})
    prediction = score_by_column.get("predicted_kwh", {})
    input_scores = [
        score
        for score in scores
        if score["column"] not in {"kWhDelivered", "predicted_kwh"}
    ]
    detected = any(score["psi_alert"] for score in scores)
    summary = {
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "ks_pvalue_alert_threshold": KS_ALERT_THRESHOLD,
        "drift_detected": detected,
        "target_psi_alert": bool(target.get("psi_alert", False)),
        "target_kwh_delivered": target,
        "prediction_distribution": prediction,
        "input_features_drifted": [
            score["column"] for score in input_scores if score["psi_alert"]
        ],
        "columns_scored": sum(score["status"] == "scored" for score in scores),
        "columns_with_insufficient_data": sum(
            score["status"] != "scored" for score in scores
        ),
        "scores": scores,
    }
    return scores, {"summary": summary, "snapshot": snapshot}


def score_month(
    settings: Settings,
    source: str,
    month: str,
    reference_frame: pd.DataFrame,
    model: Any,
    feature_spec: FeatureSpec,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    """Score one site-aggregated month and save data-free Evidently evidence."""
    from evcharge.orchestration.flows import _monthly_partitions

    partitions = _monthly_partitions(settings, source).get(month)
    if not partitions:
        raise ConfigError(f"No normalized partitions found for {source} {month}.")
    current_frame = pd.concat(
        [pd.read_parquet(path) for path in partitions], ignore_index=True
    )
    reference_scoring = _prediction_frame(
        reference_frame, model, feature_spec, reference=True
    )
    current_scoring = _prediction_frame(
        current_frame, model, feature_spec, reference=False
    )
    _, result = _score_metrics(reference_scoring, current_scoring)
    report = result["summary"]
    report.update(
        {
            "source": source,
            "batch_month": month,
            "batch_rows": len(current_frame),
            "reference_rows": len(reference_frame),
            "model_name": model_name,
            "model_version": model_version,
            "reference_snapshot": "drift_reference_v1",
            "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evidently_version": __import__("evidently").__version__,
        }
    )
    output_dir = settings.artifacts_dir / "drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source}_{month}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / f"{stem}.evidently.json").write_text(
        result["snapshot"].json(), encoding="utf-8"
    )
    result["snapshot"].save_html(str(output_dir / f"{stem}.html"))
    return report


def push_drift_metrics(reports: list[dict[str, Any]], source: str) -> None:
    """Replace the Pushgateway source group with the latest scored batch metrics."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    gateway_url = os.getenv("PUSHGATEWAY_URL", "http://127.0.0.1:9091").rstrip("/")
    registry = CollectorRegistry()
    metric_definitions = {
        "evcharge_drift_psi": ("PSI drift score.", lambda score: score["psi"]),
        "evcharge_drift_ks_statistic": (
            "Two-sample Kolmogorov-Smirnov statistic.",
            lambda score: score["ks_statistic"],
        ),
        "evcharge_drift_ks_pvalue": (
            "Two-sample Kolmogorov-Smirnov p-value.",
            lambda score: score["ks_pvalue"],
        ),
        "evcharge_drift_js_distance": (
            "Jensen-Shannon distance.",
            lambda score: score["js_distance"],
        ),
        "evcharge_drift_detected": (
            "One when the PSI alert threshold was exceeded.",
            lambda score: float(score["psi_alert"]),
        ),
    }
    gauges = {
        name: Gauge(name, description, ("source", "batch_month", "column"), registry=registry)
        for name, (description, _) in metric_definitions.items()
    }
    batch_rows = Gauge(
        "evcharge_drift_batch_rows",
        "Rows in the latest monthly batch scored.",
        ("source", "batch_month"),
        registry=registry,
    )
    latest_alert = Gauge(
        "evcharge_drift_latest_alert",
        "One when the latest available batch has a PSI alert.",
        ("source", "batch_month"),
        registry=registry,
    )
    latest_batch_rows = Gauge(
        "evcharge_drift_latest_batch_rows",
        "Rows in the latest available monthly batch.",
        ("source", "batch_month"),
        registry=registry,
    )
    sample_rows = Gauge(
        "evcharge_drift_non_missing_rows",
        "Non-missing sample count used to score a column.",
        ("source", "batch_month", "column", "dataset"),
        registry=registry,
    )
    missing_fraction = Gauge(
        "evcharge_drift_missing_fraction",
        "Missing value fraction for a scored column.",
        ("source", "batch_month", "column", "dataset"),
        registry=registry,
    )
    for report in reports:
        batch_month = report["batch_month"]
        batch_rows.labels(source=source, batch_month=batch_month).set(
            report["batch_rows"]
        )
        for score in report["scores"]:
            labels = {
                "source": source,
                "batch_month": batch_month,
                "column": score["column"],
            }
            sample_rows.labels(**labels, dataset="reference").set(
                score["reference_rows"]
            )
            sample_rows.labels(**labels, dataset="current").set(score["current_rows"])
            if score.get("reference_missing_fraction") is not None:
                missing_fraction.labels(**labels, dataset="reference").set(
                    score["reference_missing_fraction"]
                )
                missing_fraction.labels(**labels, dataset="current").set(
                    score["current_missing_fraction"]
                )
            for name, (_, get_value) in metric_definitions.items():
                value = get_value(score)
                if value is not None:
                    gauges[name].labels(**labels).set(value)
    latest_report = max(reports, key=lambda report: report["batch_month"])
    latest_alert.labels(
        source=source, batch_month=latest_report["batch_month"]
    ).set(float(latest_report["drift_detected"]))
    latest_batch_rows.labels(
        source=source, batch_month=latest_report["batch_month"]
    ).set(latest_report["batch_rows"])
    try:
        push_to_gateway(
            gateway_url,
            job="evcharge_drift",
            grouping_key={},
            registry=registry,
            timeout=10,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to export drift scores to Pushgateway at {gateway_url}."
        ) from exc


def score_new_months(
    settings: Settings, source: str, months: list[str]
) -> list[dict[str, Any]]:
    """Score all newly validated months with the same Production model."""
    if not months:
        return []
    from evcharge.orchestration.flows import _monthly_partitions

    reference = load_reference(settings)
    model, spec, model_name, model_version = load_production_model(settings, source)
    months_to_score = set(months)
    months_to_score.add(max(_monthly_partitions(settings, source)))
    reports = [
        score_month(
            settings,
            source,
            month,
            reference,
            model,
            spec,
            model_name,
            model_version,
        )
        for month in sorted(months_to_score)
    ]
    push_drift_metrics(reports, source)
    return reports
