"""Prefect entry points for training, monthly batch checks, and drift retraining."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from calendar import monthrange
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task
from prefect.client.schemas.schedules import CronSchedule
from prefect.settings import PREFECT_API_URL

from evcharge.config import ConfigError, Settings, get_settings, load_config

logger = logging.getLogger(__name__)
SOURCES = ("raw", "sample")
MONTHLY_CRON = "0 2 1 * *"
STATE_VERSION = 2


def _monthly_partitions(settings: Settings, source: str) -> dict[str, list[Path]]:
    root = settings.interim_dir / source / "sessions"
    batches: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("site=*/month=*/sessions.parquet")):
        month_dir = path.parent.name
        if not month_dir.startswith("month="):
            continue
        month = month_dir.removeprefix("month=")
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"Invalid monthly partition directory: {path.parent}") from exc
        batches[month].append(path)
    if not batches:
        raise ConfigError(f"No monthly partitions found in {root}.")
    return dict(sorted(batches.items()))


def _batch_fingerprint(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_monitor_state(path: Path, source: str) -> dict[str, Any]:
    if not path.exists():
        return {"version": STATE_VERSION, "source": source, "batches": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read monthly monitor state at {path}") from exc
    if (
        not isinstance(state, dict)
        or state.get("version") not in {1, STATE_VERSION}
        or state.get("source") != source
        or not isinstance(state.get("batches"), dict)
    ):
        raise RuntimeError(f"Monthly monitor state at {path} has an invalid schema.")
    state["version"] = STATE_VERSION
    return state


def _save_monitor_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_monthly_batches(settings: Settings, source: str) -> list[dict[str, Any]]:
    """Validate each new or changed month and persist its content fingerprint."""
    if source not in SOURCES:
        raise ValueError(f"Unsupported source {source!r}; expected one of {SOURCES}.")

    root = settings.interim_dir / source / "sessions"
    state_path = settings.artifacts_dir / "monitoring" / f"monthly_{source}.json"
    state = _load_monitor_state(state_path, source)
    spec = load_config("contract")
    outcomes: list[dict[str, Any]] = []

    from evcharge.validation.contract import ContractBreach, validate

    for month, files in _monthly_partitions(settings, source).items():
        fingerprint = _batch_fingerprint(files, root)
        previous = state["batches"].get(month, {})
        if previous.get("fingerprint") == fingerprint:
            logger.info("Monthly batch %s is unchanged; skipping.", month)
            continue

        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        report = validate(frame, spec, settings, f"{source}_{month}")
        if not report["passed"]:
            raise ContractBreach(
                f"Monthly data contract failed for {source} {month}: "
                f"{report['blocking_failures']}"
            )

        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state["batches"][month] = {
            "fingerprint": fingerprint,
            "checked_at_utc": checked_at,
            "rows": len(frame),
        }
        _save_monitor_state(state_path, state)
        outcomes.append(
            {
                "month": month,
                "rows": len(frame),
                "fingerprint": fingerprint,
                "checked_at_utc": checked_at,
                "status": "validated",
            }
        )
        logger.info("Validated monthly batch %s (%d rows).", month, len(frame))
    return outcomes


def _months_pending_drift_score(
    settings: Settings, source: str, state: dict[str, Any]
) -> list[str]:
    pending = []
    for month in _monthly_partitions(settings, source):
        batch = state["batches"].get(month, {})
        fingerprint = batch.get("fingerprint")
        if not fingerprint:
            raise RuntimeError(
                f"Monthly batch {month} has not been validated and fingerprinted."
            )
        report_path = settings.artifacts_dir / "drift" / f"{source}_{month}.json"
        if (
            batch.get("scored_fingerprint") != fingerprint
            or not report_path.is_file()
        ):
            pending.append(month)
    return pending


@task(name="Fetch ACN data", retries=2, retry_delay_seconds=60)
def fetch_raw_data() -> dict[str, Any]:
    from evcharge.pipeline import stage_fetch

    return stage_fetch(get_settings(), "raw")


@task(name="Normalise source data")
def normalise_source(source: str) -> dict[str, Any]:
    from evcharge.pipeline import stage_normalise

    return stage_normalise(get_settings(), source)


@task(name="Validate new monthly batches")
def validate_monthly_batches(source: str) -> list[dict[str, Any]]:
    return check_monthly_batches(get_settings(), source)


@task(name="Score and export monthly drift")
def score_and_export_drift(
    source: str, months: list[str]
) -> list[dict[str, Any]]:
    from evcharge.monitoring.drift import score_new_months

    return score_new_months(get_settings(), source, months)


@flow(name="train-and-register")
def training_flow(
    source: str = "raw",
    fetch: bool = False,
    split_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the existing contract-gated training pipeline as an orchestrated flow."""
    if source not in SOURCES:
        raise ValueError(f"Unsupported source {source!r}; expected one of {SOURCES}.")
    from evcharge.pipeline import DEFAULT_STAGES, run_pipeline, stage_train

    stages = (("fetch",) if fetch else ()) + DEFAULT_STAGES
    stage_overrides = None
    if split_override is not None:
        stage_overrides = {
            "train": lambda settings, current_source: stage_train(
                settings, current_source, split_override
            )
        }
    return run_pipeline(
        get_settings(), source, stages, stage_overrides=stage_overrides
    )


def retraining_split_for_month(month: str) -> dict[str, str]:
    """Use the previous full month for validation and alert month for holdout."""
    try:
        alert_month = datetime.strptime(month, "%Y-%m").date().replace(day=1)
    except ValueError as exc:
        raise ValueError(f"Invalid drift-alert month {month!r}; expected YYYY-MM.") from exc
    if alert_month.strftime("%Y-%m") != month:
        raise ValueError(f"Invalid drift-alert month {month!r}; expected YYYY-MM.")

    validation_last = alert_month - timedelta(days=1)
    validation_first = validation_last.replace(day=1)
    train_last = validation_first - timedelta(days=1)
    post_last = date(
        alert_month.year,
        alert_month.month,
        monthrange(alert_month.year, alert_month.month)[1],
    )
    return {
        "train_end": train_last.isoformat(),
        "validation_end": validation_last.isoformat(),
        "post_shift_start": alert_month.isoformat(),
        "post_shift_end": post_last.isoformat(),
    }


@flow(name="monthly-batch-monitor")
def monthly_monitoring_flow(
    source: str = "raw", fetch: bool = True
) -> dict[str, Any]:
    """Fetch, validate, score drift, export metrics, and retrain on the latest alert."""
    if source not in SOURCES:
        raise ValueError(f"Unsupported source {source!r}; expected one of {SOURCES}.")
    if fetch:
        fetch_raw_data()
    normalised = normalise_source(source)
    batches = validate_monthly_batches(source)
    drift_reports: list[dict[str, Any]] = []
    retraining: dict[str, Any] | None = None
    if source == "raw":
        settings = get_settings()
        state_path = (
            settings.artifacts_dir / "monitoring" / f"monthly_{source}.json"
        )
        state = _load_monitor_state(state_path, source)
        months = _months_pending_drift_score(settings, source, state)
        if months:
            drift_reports = score_and_export_drift(source, months)
            for month in months:
                state["batches"][month]["scored_fingerprint"] = state["batches"][
                    month
                ]["fingerprint"]
            _save_monitor_state(state_path, state)

        partitions = _monthly_partitions(settings, source)
        latest_available = max(partitions)
        latest_path = settings.artifacts_dir / "drift" / (
            f"{source}_{latest_available}.json"
        )
        latest_report = None
        if latest_path.is_file():
            latest_report = json.loads(latest_path.read_text(encoding="utf-8"))
        if (
            latest_report is not None
            and latest_report.get("batch_month") == latest_available
            and latest_report.get("drift_detected")
        ):
            batch_state = state["batches"].get(latest_available, {})
            fingerprint = batch_state.get("fingerprint")
            if not fingerprint:
                raise RuntimeError(
                    f"Latest batch {latest_available} has no validated fingerprint."
                )
            if batch_state.get("retrained_fingerprint") != fingerprint:
                retraining = retrain_on_drift_alert(
                    {
                        "drift_detected": True,
                        "batch_month": latest_available,
                        "scores": latest_report["scores"],
                    },
                    source=source,
                )
                state["batches"][latest_available][
                    "retrained_fingerprint"
                ] = fingerprint
                _save_monitor_state(state_path, state)
    return {
        "source": source,
        "normalise": normalised,
        "new_or_changed_batches": batches,
        "drift_reports": drift_reports,
        "retraining": retraining,
    }


@flow(name="retrain-on-drift-alert")
def retrain_on_drift_alert(
    alert: dict[str, Any], source: str = "raw"
) -> dict[str, Any]:
    """Start the contract-gated training flow only for an explicit drift alert."""
    if source not in SOURCES:
        raise ValueError(f"Unsupported source {source!r}; expected one of {SOURCES}.")
    if not isinstance(alert, dict) or type(alert.get("drift_detected")) is not bool:
        raise ValueError("Drift alert must contain a boolean 'drift_detected' field.")
    if not alert["drift_detected"]:
        logger.info("No drift alert; retraining was not started.")
        return {"triggered": False, "reason": "no drift detected"}
    batch_month = alert.get("batch_month")
    if not isinstance(batch_month, str):
        raise ValueError("A drift alert must include its 'batch_month' as YYYY-MM.")
    split = retraining_split_for_month(batch_month)

    logger.warning(
        "Drift alert received for %s; starting the training and promotion-gated flow.",
        batch_month,
    )
    result = training_flow(source=source, fetch=False, split_override=split)
    return {"triggered": True, "split": split, "training": result}


def serve_monthly_monitor() -> None:
    """Register and serve the raw-data monitor on a local Prefect server."""
    if not (os.environ.get("PREFECT_API_URL") or PREFECT_API_URL.value()):
        raise ConfigError(
            "Set PREFECT_API_URL to the local Prefect server, e.g. "
            "http://127.0.0.1:4200/api, before starting the monthly deployment."
        )
    monthly_monitoring_flow.serve(
        name="monthly-raw-batch-monitor",
        schedule=CronSchedule(cron=MONTHLY_CRON, timezone="UTC"),
        parameters={"source": "raw", "fetch": True},
        pause_on_shutdown=True,
    )
