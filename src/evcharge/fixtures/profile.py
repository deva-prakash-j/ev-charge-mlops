"""Derive an aggregate statistical profile from the licensed ACN-Data extract.

Run once by a maintainer who holds the data. The output
(configs/sample_profile.yaml) contains only aggregate statistics -- monthly
counts, distribution parameters, categorical weights -- and no individual
records, so it is safe to commit while the extract itself is not.

Usage:  python -m evcharge.fixtures.profile
"""

from __future__ import annotations

import glob
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

from evcharge.config import CONFIG_DIR, ConfigError, get_settings

logger = logging.getLogger(__name__)

PROFILE_START = "2019-07"
PROFILE_END = "2020-06"
SHIFT_MONTH = "2020-03"


def _lognormal_params(values: pd.Series) -> dict[str, float]:
    positive = values.dropna()
    positive = positive[positive > 0]
    if positive.empty:
        return {"log_mean": 0.0, "log_sd": 0.0, "min": 0.0, "max": 0.0}
    logs = np.log(positive)
    return {
        "log_mean": round(float(logs.mean()), 4),
        "log_sd": round(float(logs.std(ddof=0)), 4),
        "min": round(float(positive.min()), 3),
        "max": round(float(positive.quantile(0.999)), 3),
    }


def _delivery_relationship(group: pd.DataFrame) -> dict[str, float]:
    """Log-log fit of delivered against requested energy.

    Without this the fixture would carry correct marginals but no relationship
    between features and target, and any model trained on it would be unable to
    beat the mean baseline.
    """
    pair = group[["kWhRequested", "kWhDelivered"]].dropna()
    pair = pair[(pair["kWhRequested"] > 0) & (pair["kWhDelivered"] > 0)]
    if len(pair) < 50:
        return {"slope": 0.0, "intercept": 0.0, "residual_sd": 1.0, "n": len(pair)}

    x = np.log(pair["kWhRequested"].to_numpy())
    y = np.log(pair["kWhDelivered"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "residual_sd": round(float(residuals.std(ddof=0)), 4),
        "n": int(len(pair)),
    }


def build_profile(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    # siteID arrives mixed int/str in the raw feed; normalise for profiling only.
    frame["siteID"] = frame["siteID"].apply(lambda v: f"{int(v):04d}" if isinstance(v, (int, np.integer)) else str(v))
    frame["ym"] = frame["connectionTimeLocal"].dt.strftime("%Y-%m")
    window = frame[frame["ym"].between(PROFILE_START, PROFILE_END)]

    sites: dict[str, Any] = {}
    for site, group in window.groupby("siteID"):
        monthly = group.groupby("ym").size()
        pre = group[group["ym"] < SHIFT_MONTH]
        post = group[group["ym"] >= SHIFT_MONTH]
        sites[site] = {
            "monthly_counts": {m: int(n) for m, n in monthly.items()},
            "stations": int(group["stationID"].nunique()),
            "clusters": sorted({str(c) for c in group["clusterID"].dropna().unique()}),
            "timezone": str(group["timezone"].mode().iat[0]),
            "user_inputs_present_rate": round(float(group["kWhRequested"].notna().mean()), 4),
            "kwh_delivered": {
                "pre_shift": _lognormal_params(pre["kWhDelivered"]),
                "post_shift": _lognormal_params(post["kWhDelivered"]),
            },
            "connected_hours": {
                "pre_shift": _lognormal_params(pre["connectedHours"]),
                "post_shift": _lognormal_params(post["connectedHours"]),
            },
            "kwh_requested": _lognormal_params(group["kWhRequested"]),
            "minutes_available": _lognormal_params(group["minutesAvailable"]),
            "delivery_given_request": {
                "pre_shift": _delivery_relationship(pre),
                "post_shift": _delivery_relationship(post),
            },
        }

    hours = window["connectionTimeLocal"].dt.hour.value_counts().sort_index()
    dow = window["connectionTimeLocal"].dt.dayofweek.value_counts().sort_index()

    return {
        "_comment": (
            "Aggregate statistics derived from the licensed ACN-Data extract. "
            "Contains no individual records. Used to generate the synthetic "
            "fixture in data/sample/ so the pipeline runs without credentials."
        ),
        "derived_from": {
            "source": "ACN-Data",
            "window": f"{PROFILE_START}..{PROFILE_END}",
            "shift_month": SHIFT_MONTH,
            "sessions_profiled": int(len(window)),
        },
        "arrival_hour_weights": {int(h): int(n) for h, n in hours.items()},
        "day_of_week_weights": {int(d): int(n) for d, n in dow.items()},
        "wh_per_mile_choices": sorted(
            {int(v) for v in window["WhPerMile"].dropna().unique()}
        )[:12],
        "sites": sites,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    pattern = str(settings.interim_dir / "raw" / "sessions" / "**" / "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(
            "[FAIL] No normalised sessions found. Run:\n"
            "       python -m evcharge.ingest.normalise --source raw",
            file=sys.stderr,
        )
        return 2

    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    profile = build_profile(frame)

    out = CONFIG_DIR / "sample_profile.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(profile, fh, sort_keys=False, default_flow_style=False)

    print(f"[ok]   profiled {profile['derived_from']['sessions_profiled']:,} sessions")
    print(f"[ok]   sites: {list(profile['sites'])}")
    print(f"[ok]   profile -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
