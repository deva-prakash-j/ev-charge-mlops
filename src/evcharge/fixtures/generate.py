"""Generate the committed synthetic sample fixture.

Reads only configs/sample_profile.yaml (aggregate statistics), so it runs
without the licensed extract and without an API token. Output is deterministic
for a given seed.

Deliberately excluded from the fixture: userID and every other pseudonymous or
record-level identifier. Synthetic sessionIDs are generated instead.

Usage:  python -m evcharge.fixtures.generate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from evcharge.config import ConfigError, get_settings, load_config

logger = logging.getLogger(__name__)

DEFAULT_SEED = 20260407
HTTP_DATE_FMT = "%a, %d %b %Y %H:%M:%S GMT"

# Below this, the fitted delivered-vs-requested relationship is not trustworthy
# and the marginal distribution is used instead.
MIN_FIT_OBSERVATIONS = 50


def _draw_lognormal(rng: np.random.Generator, params: dict[str, float], size: int) -> np.ndarray:
    if size == 0 or params["log_sd"] == 0:
        return np.full(size, params.get("min", 0.0))
    values = rng.lognormal(params["log_mean"], params["log_sd"], size)
    return np.clip(values, params["min"], params["max"])


def _month_bounds(month: str) -> tuple[datetime, int]:
    start = datetime.strptime(month, "%Y-%m")
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start, (next_month - start).days


def _sample_timestamps(
    rng: np.random.Generator,
    month: str,
    count: int,
    hour_weights: dict[int, int],
    dow_weights: dict[int, int],
) -> list[datetime]:
    start, days_in_month = _month_bounds(month)

    hours = np.array(sorted(hour_weights))
    hour_p = np.array([hour_weights[h] for h in hours], dtype=float)
    hour_p /= hour_p.sum()

    dows = np.array(sorted(dow_weights))
    dow_p = np.array([dow_weights[d] for d in dows], dtype=float)
    dow_p /= dow_p.sum()

    day_choices = np.arange(days_in_month)
    day_p = np.array(
        [dow_p[list(dows).index((start + timedelta(days=int(d))).weekday())] for d in day_choices]
    )
    day_p /= day_p.sum()

    days = rng.choice(day_choices, size=count, p=day_p)
    picked_hours = rng.choice(hours, size=count, p=hour_p)
    minutes = rng.integers(0, 60, size=count)
    seconds = rng.integers(0, 60, size=count)

    return [
        start
        + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days, picked_hours, minutes, seconds)
    ]


def generate(profile: dict[str, Any], seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shift_month = profile["derived_from"]["shift_month"]
    hour_weights = {int(k): v for k, v in profile["arrival_hour_weights"].items()}
    dow_weights = {int(k): v for k, v in profile["day_of_week_weights"].items()}
    wh_choices = profile["wh_per_mile_choices"]

    rows: list[dict[str, Any]] = []

    for site, spec in profile["sites"].items():
        clusters = spec["clusters"] or ["0001"]
        stations = [f"{site}-S{i:03d}" for i in range(spec["stations"])]
        tz = spec["timezone"]
        zone = ZoneInfo(tz)
        present_rate = spec["user_inputs_present_rate"]

        for month, real_count in spec["monthly_counts"].items():
            # Scale down but keep the month-to-month shape, which carries the shift.
            count = max(5, round(real_count * 0.12))
            regime = "post_shift" if month >= shift_month else "pre_shift"

            connections = _sample_timestamps(rng, month, count, hour_weights, dow_weights)
            marginal_kwh = _draw_lognormal(rng, spec["kwh_delivered"][regime], count)
            hours = _draw_lognormal(rng, spec["connected_hours"][regime], count)
            requested = _draw_lognormal(rng, spec["kwh_requested"], count)
            minutes = _draw_lognormal(rng, spec["minutes_available"], count)
            has_inputs = rng.random(count) < present_rate

            # Where a driver request exists and the site/regime had enough
            # observations to fit a relationship, derive delivered energy from
            # the request so the fixture carries learnable signal rather than
            # only correct marginals. Otherwise fall back to the marginal.
            relationship = spec["delivery_given_request"][regime]
            bounds = spec["kwh_delivered"][regime]
            if relationship["n"] >= MIN_FIT_OBSERVATIONS:
                derived = np.exp(
                    relationship["intercept"]
                    + relationship["slope"] * np.log(requested)
                    + rng.normal(0.0, relationship["residual_sd"], count)
                )
                kwh = np.where(
                    has_inputs,
                    np.clip(derived, bounds["min"], bounds["max"]),
                    marginal_kwh,
                )
            else:
                kwh = marginal_kwh

            for i in range(count):
                # Arrival hours are sampled in site-local time, so localise before
                # converting to the UTC wire format the API uses.
                connect = connections[i].replace(tzinfo=zone).astimezone(timezone.utc)
                disconnect = connect + timedelta(hours=float(hours[i]))
                inputs_present = bool(has_inputs[i])
                rows.append(
                    {
                        "sessionID": f"SYN-{site}-{month}-{i:05d}",
                        "siteID": site,
                        "clusterID": str(rng.choice(clusters)),
                        "stationID": str(rng.choice(stations)),
                        "spaceID": f"SP-{rng.integers(1, spec['stations'] + 1):03d}",
                        "timezone": tz,
                        "connectionTime": connect.strftime(HTTP_DATE_FMT),
                        "disconnectTime": disconnect.strftime(HTTP_DATE_FMT),
                        "doneChargingTime": None,
                        "kWhDelivered": round(float(kwh[i]), 3),
                        "userInputs": (
                            [
                                {
                                    "kWhRequested": round(float(requested[i]), 2),
                                    "milesRequested": int(requested[i] * 3),
                                    "minutesAvailable": int(minutes[i]),
                                    "WhPerMile": int(rng.choice(wh_choices)),
                                    "requestedDeparture": (
                                        connect + timedelta(minutes=float(minutes[i]))
                                    ).strftime(HTTP_DATE_FMT),
                                    "paymentRequired": True,
                                    "modifiedAt": connect.strftime(HTTP_DATE_FMT),
                                }
                            ]
                            if inputs_present
                            else []
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    return frame.sort_values("connectionTime").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic sample fixture.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        settings = get_settings()
        profile = load_config("sample_profile")
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    frame = generate(profile, args.seed)

    settings.sample_dir.mkdir(parents=True, exist_ok=True)
    target = settings.sample_dir / "sessions_sample.jsonl"
    digest = hashlib.sha256()
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for record in frame.to_dict(orient="records"):
            line = json.dumps(record, separators=(",", ":"), sort_keys=True, default=str)
            fh.write(line + "\n")
            digest.update(line.encode("utf-8"))

    manifest = {
        "source": "SYNTHETIC — generated from aggregate statistics, not real records",
        "generator": "evcharge.fixtures.generate",
        "seed": args.seed,
        "profile_window": profile["derived_from"]["window"],
        "shift_month": profile["derived_from"]["shift_month"],
        "sessions": len(frame),
        "sha256": digest.hexdigest(),
        "contains_pii": False,
        "contains_real_records": False,
        "note": (
            "Real ACN-Data records are not redistributable. This fixture reproduces "
            "the schema and the aggregate distributions so the pipeline is runnable "
            "without credentials. Headline results in the report use the licensed extract."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (settings.sample_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[ok]   generated {len(frame):,} synthetic sessions")
    print(f"[ok]   months: {frame['connectionTime'].count()} rows across "
          f"{frame['siteID'].nunique()} sites")
    print(f"[ok]   sha256: {digest.hexdigest()[:16]}...")
    print(f"[ok]   fixture -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
