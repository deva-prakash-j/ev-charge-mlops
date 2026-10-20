"""Great Expectations data contract for normalised charging sessions.

Two layers:
  * schema/shape expectations -- a breach blocks the pipeline;
  * row-level plausibility rules -- offending rows are quarantined with a
    reason and the run continues on the clean remainder.

Quarantining rather than dropping keeps bad data visible: the real ACN-Data
feed contains 245-hour sessions, and silently discarding them would hide a
genuine upstream defect.

Usage:  python -m evcharge.validation.contract --source sample
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from evcharge.config import ConfigError, Settings, get_settings, load_config

logger = logging.getLogger(__name__)


class ContractBreach(RuntimeError):
    """Raised when a blocking expectation fails."""


OPS = {
    "gt": lambda s, v: s > v,
    "ge": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "le": lambda s, v: s <= v,
    "eq": lambda s, v: s == v,
    "isnull": lambda s, _v: s.isna(),
}


def apply_quarantine(
    frame: pd.DataFrame, rules: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the batch into clean rows and quarantined rows with reasons."""
    reasons = pd.Series([""] * len(frame), index=frame.index, dtype=object)

    for rule in rules:
        column, op = rule["column"], rule["op"]
        if column not in frame.columns:
            logger.warning("quarantine rule skipped, no column %r", column)
            continue
        if op not in OPS:
            raise ConfigError(f"Unknown quarantine op {op!r} for column {column!r}")

        mask = OPS[op](frame[column], rule.get("value")).fillna(False)
        if mask.any():
            reasons.loc[mask] = reasons.loc[mask].str.cat(
                [rule["reason"]] * int(mask.sum()), sep="; "
            ).str.lstrip("; ")

    flagged = reasons != ""
    quarantined = frame.loc[flagged].copy()
    quarantined["quarantine_reason"] = reasons.loc[flagged]
    return frame.loc[~flagged].copy(), quarantined


def run_expectations(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the GE suite against the batch."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger("great_expectations").setLevel(logging.ERROR)

    import great_expectations as gx
    from great_expectations.data_context.types.base import ProgressBarsConfig

    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(
        globally=False, metric_calculations=False
    )
    datasource = context.sources.add_pandas("evcharge")
    asset = datasource.add_dataframe_asset("sessions")
    context.add_or_update_expectation_suite("sessions_contract")
    validator = context.get_validator(
        batch_request=asset.build_batch_request(dataframe=frame),
        expectation_suite_name="sessions_contract",
    )

    severities: dict[int, str] = {}
    for index, expectation in enumerate(spec["expectations"]):
        method = getattr(validator, expectation["type"], None)
        if method is None:
            raise ConfigError(f"Unknown expectation type {expectation['type']!r}")
        method(**expectation["kwargs"])
        severities[index] = expectation.get("severity", "block")

    result = validator.validate()

    checks = []
    for index, item in enumerate(result.results):
        kwargs = dict(item.expectation_config.kwargs)
        kwargs.pop("batch_id", None)
        checks.append(
            {
                "expectation": item.expectation_config.expectation_type,
                "column": kwargs.get("column"),
                "kwargs": kwargs,
                "severity": severities.get(index, "block"),
                "success": bool(item.success),
                "unexpected_count": item.result.get("unexpected_count"),
                "unexpected_percent": item.result.get("unexpected_percent"),
            }
        )

    custom_checks = []
    station_domains = spec.get("station_id_domains", {})
    if station_domains:
        sites = frame["siteID"].astype("string")
        stations = frame["stationID"].astype("string")
        valid_station = pd.Series(False, index=frame.index)
        for site, pattern in station_domains.items():
            valid_station |= sites.eq(site).fillna(False) & stations.str.fullmatch(
                pattern, na=False
            ).fillna(False)
        unexpected_count = int((~valid_station).sum())
        custom_checks.append(
            {
                "expectation": "expect_station_id_in_site_domain",
                "column": "stationID",
                "kwargs": {"site_domains": station_domains},
                "severity": spec.get("station_id_domain_severity", "block"),
                "success": unexpected_count == 0,
                "unexpected_count": unexpected_count,
                "unexpected_percent": (
                    100.0 * unexpected_count / len(frame) if len(frame) else 0.0
                ),
            }
        )

    energy_rule = spec.get("declared_energy_plausibility")
    if energy_rule:
        requested = pd.to_numeric(frame["kWhRequested"], errors="coerce")
        delivered = pd.to_numeric(frame["kWhDelivered"], errors="coerce")
        eligible = requested.notna() & delivered.notna()
        plausible = (requested > 0) & (
            delivered
            <= requested * energy_rule["max_delivered_to_requested_ratio"]
        )
        plausible |= (requested == 0) & (delivered == 0)
        unexpected_count = int((eligible & ~plausible).sum())
        eligible_count = int(eligible.sum())
        plausible_fraction = (
            (eligible_count - unexpected_count) / eligible_count
            if eligible_count
            else 0.0
        )
        custom_checks.append(
            {
                "expectation": "expect_declared_energy_ratio_to_be_plausible",
                "column": "kWhRequested",
                "kwargs": {
                    "max_delivered_to_requested_ratio":
                        energy_rule["max_delivered_to_requested_ratio"],
                    "minimum_plausible_fraction":
                        energy_rule["minimum_plausible_fraction"],
                    "excluded_missing_requests": True,
                },
                "severity": energy_rule.get("severity", "block"),
                "success": (
                    eligible_count > 0
                    and plausible_fraction >= energy_rule["minimum_plausible_fraction"]
                ),
                "unexpected_count": unexpected_count,
                "unexpected_percent": (
                    100.0 * unexpected_count / eligible_count
                    if eligible_count
                    else 100.0
                ),
            }
        )

    checks.extend(custom_checks)
    result_statistics = dict(result.statistics)
    result_statistics["evaluated_expectations"] += len(custom_checks)
    result_statistics["successful_expectations"] += sum(
        bool(check["success"]) for check in custom_checks
    )
    result_statistics["unsuccessful_expectations"] += sum(
        not bool(check["success"]) for check in custom_checks
    )
    custom_success = all(check["success"] for check in custom_checks)

    return {
        "success": bool(result.success and custom_success),
        "statistics": result_statistics,
        "checks": checks,
    }


def load_normalised(settings: Settings, source: str) -> pd.DataFrame:
    pattern = str(settings.interim_dir / source / "sessions" / "**" / "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise ConfigError(
            f"No normalised sessions for source {source!r}. Run: "
            f"python -m evcharge.ingest.normalise --source {source}"
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def validate(
    frame: pd.DataFrame, spec: dict[str, Any], settings: Settings, source: str
) -> dict[str, Any]:
    missing = [c for c in spec["required_columns"] if c not in frame.columns]
    if missing:
        raise ContractBreach(f"Required columns missing: {missing}")

    clean, quarantined = apply_quarantine(frame, spec["quarantine_rules"])
    outcome = run_expectations(clean, spec)

    blocking = [c for c in outcome["checks"] if not c["success"] and c["severity"] == "block"]

    report = {
        "source": source,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_in": len(frame),
        "rows_clean": len(clean),
        "rows_quarantined": len(quarantined),
        "quarantine_reasons": (
            quarantined["quarantine_reason"].value_counts().to_dict()
            if len(quarantined)
            else {}
        ),
        "expectations": outcome,
        "blocking_failures": blocking,
        "passed": not blocking,
    }

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / f"contract_{source}.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )

    if len(quarantined):
        out_dir = settings.data_dir / "quarantine" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        quarantined.to_parquet(out_dir / "quarantined.parquet", index=False)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the session data contract.")
    parser.add_argument("--source", choices=("sample", "raw"), default="sample")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    try:
        settings = get_settings()
        spec = load_config("contract")
        frame = load_normalised(settings, args.source)
        report = validate(frame, spec, settings, args.source)
    except (ConfigError, ContractBreach) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    stats = report["expectations"]["statistics"]
    print(f"\n[ok]   source:        {args.source}")
    print(f"[ok]   rows in:       {report['rows_in']:,}")
    print(f"[ok]   rows clean:    {report['rows_clean']:,}")
    print(f"[{'WARN' if report['rows_quarantined'] else 'ok'}]   quarantined:   "
          f"{report['rows_quarantined']:,}")
    for reason, count in report["quarantine_reasons"].items():
        print(f"         - {reason}: {count}")
    print(f"[ok]   expectations:  {stats['successful_expectations']}/"
          f"{stats['evaluated_expectations']} passed")

    if not report["passed"]:
        print("\n[FAIL] Blocking contract failures:", file=sys.stderr)
        for check in report["blocking_failures"]:
            print(f"         {check['expectation']}({check['column']}) "
                  f"unexpected={check['unexpected_count']}", file=sys.stderr)
        return 1

    print("\n[ok]   data contract satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
