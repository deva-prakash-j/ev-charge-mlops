"""Reproduce monthly-batch drift, leave-one-month-out, and injected-shift checks."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from evcharge.config import DRIFT_SOURCES, get_settings
from evcharge.monitoring.drift import (
    PSI_ALERT_THRESHOLD,
    _prediction_frame,
    _score_metrics,
    load_production_model,
    load_reference,
)
from evcharge.orchestration.flows import _monthly_partitions


def _target_score(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, Any]:
    scores, _ = _score_metrics(reference, current)
    return next(score for score in scores if score["column"] == "kWhDelivered")


def evaluate(months: list[str], source: str = "raw") -> dict[str, Any]:
    settings = get_settings()
    reference = load_reference(settings, source)
    model, feature_spec, model_name, model_version = load_production_model(
        settings, source
    )
    reference_scoring = _prediction_frame(
        reference, model, feature_spec, reference=True
    )
    partitions = _monthly_partitions(settings, source)
    reference_months = pd.to_datetime(reference["connectionTimeLocal"]).dt.strftime(
        "%Y-%m"
    )
    reference_counts = (
        reference.assign(_month=reference_months)
        .groupby(["siteID", "_month"])
        .size()
        .unstack(fill_value=0)
    )
    reference_site_medians = reference_counts.median(axis=1)

    real_batches: dict[str, Any] = {}
    for month in months:
        files = partitions.get(month)
        if not files:
            raise ValueError(f"No {source} monthly partitions are available for {month}.")
        batch = pd.concat(
            [pd.read_parquet(path) for path in files], ignore_index=True
        )
        scoring = _prediction_frame(batch, model, feature_spec, reference=False)
        batch_site_counts = batch.groupby("siteID").size()
        site_volume = {}
        for site, baseline_count in reference_site_medians.items():
            observed_count = int(batch_site_counts.get(site, 0))
            site_volume[str(site)] = {
                "batch_rows": observed_count,
                "reference_monthly_median_rows": float(baseline_count),
                "change_fraction": (
                    observed_count / baseline_count - 1
                    if baseline_count > 0
                    else None
                ),
            }
        real_batches[month] = {
            "batch_rows": len(batch),
            "site_volume_vs_reference_median": site_volume,
            "target": _target_score(reference_scoring, scoring),
        }

    reference_dates = pd.to_datetime(reference["connectionTimeLocal"])
    stable_months: dict[str, Any] = {}
    for month in sorted(reference_months.unique()):
        is_month = reference_dates.dt.strftime("%Y-%m") == month
        comparison_reference = reference.loc[~is_month].reset_index(drop=True)
        comparison_current = reference.loc[is_month].reset_index(drop=True)
        ref_features = _prediction_frame(
            comparison_reference, model, feature_spec, reference=True
        )
        current_features = _prediction_frame(
            comparison_current, model, feature_spec, reference=True
        )
        stable_months[month] = {
            "batch_rows": len(comparison_current),
            "target": _target_score(ref_features, current_features),
        }

    sensitivity: list[dict[str, Any]] = []
    for factor in (1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
        shifted = reference_scoring.copy()
        shifted["kWhDelivered"] = shifted["kWhDelivered"] * factor
        score = _target_score(reference_scoring, shifted)
        sensitivity.append(
            {
                "energy_multiplier": factor,
                "psi": score["psi"],
                "psi_alert": score["psi_alert"],
                "reference_rows": score["reference_rows"],
                "current_rows": score["current_rows"],
            }
        )

    hypothesis_months = [
        month for month in partitions if "2021-05" <= month <= "2021-09"
    ]
    if not hypothesis_months:
        raise ValueError(f"{source} monthly batches for the H2/H3 2021 window are missing.")
    later = pd.concat(
        [
            pd.read_parquet(path)
            for month in hypothesis_months
            for path in partitions[month]
        ],
        ignore_index=True,
    )
    reference_requested = reference[
        pd.to_numeric(reference["kWhRequested"], errors="coerce") > 0
    ].copy()
    later_requested = later[
        pd.to_numeric(later["kWhRequested"], errors="coerce") > 0
    ].copy()
    h2_site_comparison = {}
    for site in sorted(set(reference_requested["siteID"]) & set(later_requested["siteID"])):
        ref_site = reference.loc[reference["siteID"] == site]
        later_site = later.loc[later["siteID"] == site]
        ref_values = reference_requested.loc[
            reference_requested["siteID"] == site, "kWhRequested"
        ].dropna()
        current_values = later_requested.loc[
            later_requested["siteID"] == site, "kWhRequested"
        ].dropna()
        reference_p90 = float(ref_values.quantile(0.9))
        current_p90 = float(current_values.quantile(0.9))
        h2_site_comparison[str(site)] = {
            "reference_total_rows": len(ref_site),
            "reference_positive_request_rows": len(ref_values),
            "reference_request_coverage": len(ref_values) / len(ref_site),
            "later_total_rows": len(later_site),
            "later_positive_request_rows": len(current_values),
            "later_request_coverage": len(current_values) / len(later_site),
            "reference_p90_kwh": reference_p90,
            "later_p90_kwh": current_p90,
            "change_fraction": current_p90 / reference_p90 - 1,
        }

    reference_gaps = pd.to_numeric(
        reference_requested["kWhRequested"], errors="coerce"
    ) - pd.to_numeric(reference_requested["kWhDelivered"], errors="coerce")
    later_gaps = pd.to_numeric(
        later_requested["kWhRequested"], errors="coerce"
    ) - pd.to_numeric(later_requested["kWhDelivered"], errors="coerce")
    gap_scores, _ = _score_metrics(
        pd.DataFrame({"declared_delivered_gap_kwh": reference_gaps}),
        pd.DataFrame({"declared_delivered_gap_kwh": later_gaps}),
    )
    gap_score = gap_scores[0]
    reference_gap_median = float(reference_gaps.median())
    later_gap_median = float(later_gaps.median())

    manifest = json.loads(
        (settings.reference_dir(source) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    false_alarms = sum(
        item["target"]["psi_alert"] for item in stable_months.values()
    )
    return {
        "source": source,
        "model_name": model_name,
        "model_version": model_version,
        "reference_sha256": manifest["sha256"],
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "real_batches": real_batches,
        "stable_leave_one_month_out": {
            "false_alarms": false_alarms,
            "months_tested": len(stable_months),
            "months": stable_months,
        },
        "controlled_energy_shift_sensitivity": sensitivity,
        "fleet_mix_hypothesis_2021_05_to_2021_09": {
            "measure": "90th percentile of positive kWhRequested by shared site",
            "sites": h2_site_comparison,
        },
        "declared_delivered_gap_hypothesis_2021_05_to_2021_09": {
            "reference_positive_request_rows": len(reference_gaps),
            "later_positive_request_rows": len(later_gaps),
            "reference_median_gap_kwh": reference_gap_median,
            "later_median_gap_kwh": later_gap_median,
            "absolute_median_change_kwh": abs(later_gap_median - reference_gap_median),
            "distribution_score": gap_score,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=DRIFT_SOURCES,
        default="raw",
        help="'synthetic' runs every check on the committed synthetic dataset.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=["2020-03", "2020-04"],
        help="Monthly batches to score against the frozen reference.",
    )
    args = parser.parse_args()

    report = evaluate(args.months, args.source)
    suffix = "" if args.source == "raw" else f"_{args.source}"
    output = get_settings().artifacts_dir / "drift" / f"detector_evaluation{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Evaluation report saved to {output}")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
