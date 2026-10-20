"""Full-period synthetic dataset for credential-free drift evaluation.

The small `sample` fixture keeps CI fast but covers only 2019-07..2020-06 at
12% volume, too little for the drift reference, monthly monitoring, detector
evaluation and retraining-recovery checks. This module produces a separate
`synthetic` source with the licensed extract's full period and monthly volumes.

Two steps, kept apart so the committed data never depends on licensed records:

    python -m evcharge.fixtures.synthetic profile    # maintainer, needs the extract
    python -m evcharge.fixtures.synthetic generate   # anyone, reads the profile only

`profile` writes configs/synthetic_profile.yaml: monthly session and request
counts and, per site and month (falling back to quarter, then regime), interior
quantiles plus regression and copula parameters. Each fitted block requires at
least MIN_OBSERVATIONS sessions and no extreme order statistic is stored, so no
statistic describes an individual. `generate` draws every session from those
parameters with a fixed seed. It emits no real identifiers, timestamps or
measurements.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import logging
import sys
from datetime import timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, rankdata

from evcharge.config import CONFIG_DIR, ConfigError, get_settings, load_config
from evcharge.fixtures.generate import HTTP_DATE_FMT, _sample_timestamps

logger = logging.getLogger(__name__)

DEFAULT_SEED = 20261020
PROFILE_NAME = "synthetic_profile"
DATA_FILE = "sessions_synthetic.jsonl.gz"
MIN_OBSERVATIONS = 50
REGIMES: dict[str, tuple[str, str]] = {
    "pre_shift": ("2018-04", "2020-02"),
    "shutdown": ("2020-03", "2020-12"),
    "reopening": ("2021-01", "2021-09"),
}
MAX_DELIVERED_TO_REQUESTED = 3.0
# Interior probabilities only: extreme order statistics would describe single sessions.
QUANTILE_PROBS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995,
)


def regime_for(month: str) -> str:
    for name, (start, end) in REGIMES.items():
        if start <= month <= end:
            return name
    raise ValueError(f"Month {month} is outside the synthetic regimes {REGIMES}.")


def quarter_for(month: str) -> str:
    """Calendar quarter, split where a regime boundary falls inside it."""
    year, number = month.split("-")
    return f"{year}Q{(int(number) - 1) // 3 + 1}/{regime_for(month)}"


def _round(values: Any, digits: int = 5) -> Any:
    array = np.asarray(values, dtype=float)
    return np.round(array, digits).tolist()


def _bounds(values: pd.Series) -> list[float]:
    positive = values[values > 0].dropna()
    low, high = positive.quantile([QUANTILE_PROBS[0], QUANTILE_PROBS[-1]])
    return [round(float(low), 3), round(float(high), 3)]


def _fit(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    return coef, response - design @ coef


def _quantiles(values: np.ndarray) -> list[float]:
    return _round(np.quantile(values, QUANTILE_PROBS))


def _copula_corr(columns: list[np.ndarray]) -> list[list[float]]:
    """Correlation of normal scores, so marginals can be drawn from quantiles."""
    scores = [norm.ppf((rankdata(column) - 0.5) / len(column)) for column in columns]
    return _round(np.corrcoef(np.vstack(scores)))


def _block_stats(group: pd.DataFrame) -> dict[str, Any]:
    """Aggregate parameters for one slice; blocks below the floor are None.

    Marginals are stored as quantiles and joined with a Gaussian copula, which
    keeps skew and multimodality that a lognormal fit would flatten.
    """
    stats: dict[str, Any] = {
        "n": int(len(group)),
        "user_inputs_present_rate": round(float(group["kWhRequested"].notna().mean()), 4),
        "bounds": {
            "kwh_delivered": _bounds(group["kWhDelivered"]),
            "connected_hours": _bounds(group["connectedHours"]),
        },
        "with_inputs": None,
        "without_inputs": None,
    }

    inputs = group[
        (group["kWhRequested"] > 0) & (group["minutesAvailable"] > 0)
    ]
    if len(inputs) >= MIN_OBSERVATIONS:
        log_req = np.log(inputs["kWhRequested"].to_numpy(dtype=float))
        log_min = np.log(inputs["minutesAvailable"].to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(inputs)), log_req, log_min])
        energy_coef, energy_res = _fit(
            design, np.log(inputs["kWhDelivered"].to_numpy(dtype=float))
        )
        dwell_coef, dwell_res = _fit(
            design, np.log(inputs["connectedHours"].to_numpy(dtype=float))
        )
        stats["with_inputs"] = {
            "n": int(len(inputs)),
            "log_kwh_requested_q": _quantiles(log_req),
            "log_minutes_available_q": _quantiles(log_min),
            "energy_coef": _round(energy_coef),
            "dwell_coef": _round(dwell_coef),
            "energy_residual_q": _quantiles(energy_res),
            "dwell_residual_q": _quantiles(dwell_res),
            "copula_corr": _copula_corr([log_req, log_min, energy_res, dwell_res]),
        }

    no_inputs = group[group["kWhRequested"].isna()]
    if len(no_inputs) >= MIN_OBSERVATIONS:
        log_kwh = np.log(no_inputs["kWhDelivered"].to_numpy(dtype=float))
        log_hours = np.log(no_inputs["connectedHours"].to_numpy(dtype=float))
        stats["without_inputs"] = {
            "n": int(len(no_inputs)),
            "log_kwh_delivered_q": _quantiles(log_kwh),
            "log_connected_hours_q": _quantiles(log_hours),
            "copula_corr": _copula_corr([log_kwh, log_hours]),
        }
    return stats


def _weights(values: pd.Series) -> dict[int, int]:
    return {int(key): int(count) for key, count in values.value_counts().sort_index().items()}


def build_profile(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    frame["ym"] = pd.to_datetime(frame["connectionTimeLocal"]).dt.strftime("%Y-%m")
    first, last = min(r[0] for r in REGIMES.values()), max(r[1] for r in REGIMES.values())
    frame = frame[frame["ym"].between(first, last)]
    counts = frame.groupby(["siteID", "ym"]).size()
    # Exact per-month request volume: a Bernoulli draw at sub-1% coverage can empty
    # a month, which the data contract rightly blocks.
    request_counts = frame.groupby(["siteID", "ym"])["kWhRequested"].agg(
        lambda values: int((values > 0).sum())
    )

    # Fitted blocks use plausible sessions only; volumes keep every row.
    usable = frame[
        frame["kWhDelivered"].between(0, 200, inclusive="neither")
        & frame["connectedHours"].between(0, 168, inclusive="neither")
    ].copy()
    usable["regime"] = usable["ym"].map(regime_for)
    usable["quarter"] = usable["ym"].map(quarter_for)
    local = pd.to_datetime(usable["connectionTimeLocal"])
    usable["hour"] = local.dt.hour
    usable["dow"] = local.dt.dayofweek
    usable["kw"] = usable["kWhDelivered"] / usable["connectedHours"]

    def blocks(group: pd.DataFrame, key: str) -> dict[str, Any]:
        return {
            str(label): _block_stats(part)
            for label, part in group.groupby(key)
            if len(part) >= MIN_OBSERVATIONS
        }

    sites: dict[str, Any] = {}
    for site, group in usable.groupby("siteID"):
        site_counts = counts.loc[site]
        site_requests = request_counts.loc[site]
        sites[str(site)] = {
            "timezone": str(group["timezone"].mode().iat[0]),
            "stations": int(group["stationID"].nunique()),
            "clusters": sorted({str(c) for c in group["clusterID"].dropna().unique()}),
            "max_kw": round(float(group["kw"].quantile(QUANTILE_PROBS[-1])), 3),
            "monthly_counts": {str(m): int(n) for m, n in site_counts.items() if n > 0},
            "monthly_request_counts": {
                str(m): int(n) for m, n in site_requests.items() if n > 0
            },
            "pooled": _block_stats(group),
            "regimes": blocks(group, "regime"),
            "quarters": blocks(group, "quarter"),
            "months": blocks(group, "ym"),
        }

    wh = usable["WhPerMile"].dropna().astype(int).value_counts().head(12).sort_index()
    return {
        "_comment": (
            "Aggregate statistics derived from the licensed ACN-Data extract: monthly "
            "counts, interior quantiles and regression/copula parameters per site and "
            f"month, quarter or regime, each fitted on at least {MIN_OBSERVATIONS} "
            "sessions. Contains no individual records. Used to generate data/synthetic/."
        ),
        "derived_from": {
            "source": "ACN-Data",
            "window": f"{first}..{last}",
            "sessions_profiled": int(len(frame)),
            "min_observations": MIN_OBSERVATIONS,
        },
        "regimes": {name: list(bounds) for name, bounds in REGIMES.items()},
        "quantile_probs": list(QUANTILE_PROBS),
        "arrival_hour_weights": {
            regime: _weights(part["hour"]) for regime, part in usable.groupby("regime")
        },
        "day_of_week_weights": {
            regime: _weights(part["dow"]) for regime, part in usable.groupby("regime")
        },
        "wh_per_mile_weights": {int(k): int(v) for k, v in wh.items()},
        # All-site fallback when one site's regime has too few sessions for a block.
        "all_sites_by_regime": blocks(usable, "regime"),
        "sites": sites,
    }


def _pick(profile: dict[str, Any], spec: dict[str, Any], month: str, block: str) -> Any:
    """Most specific block with enough sessions: site month, quarter, regime, all sites."""
    regime = regime_for(month)
    for stats in (
        spec["months"].get(month),
        spec["quarters"].get(quarter_for(month)),
        spec["regimes"].get(regime),
        profile["all_sites_by_regime"].get(regime),
        spec["pooled"],
    ):
        if stats is not None and stats.get(block) is not None:
            return stats[block]
    raise ConfigError(f"Synthetic profile has no '{block}' block for {month}.")


def _draw_copula(
    rng: np.random.Generator,
    corr: list[list[float]],
    quantiles: list[list[float] | None],
    probs: np.ndarray,
    size: int,
) -> np.ndarray:
    """Correlated draws: normal scores -> uniforms -> each column's quantile function."""
    dims = len(corr)
    if size == 0:
        return np.empty((0, dims))
    scores = rng.multivariate_normal(np.zeros(dims), corr, size=size, method="eigh")
    uniforms = norm.cdf(scores)
    return np.column_stack(
        [np.interp(uniforms[:, i], probs, quantiles[i]) for i in range(dims)]
    )


def generate(profile: dict[str, Any], seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = np.asarray(profile["quantile_probs"], dtype=float)
    wh_values = np.array([int(v) for v in profile["wh_per_mile_weights"]])
    wh_p = np.array(list(profile["wh_per_mile_weights"].values()), dtype=float)
    wh_p /= wh_p.sum()
    rows: list[dict[str, Any]] = []

    for site, spec in profile["sites"].items():
        zone = ZoneInfo(spec["timezone"])
        stations = [f"{site}-S{i:03d}" for i in range(spec["stations"])]
        clusters = spec["clusters"] or ["0001"]

        for month, count in spec["monthly_counts"].items():
            regime = regime_for(month)
            hour_weights = {int(k): v for k, v in profile["arrival_hour_weights"][regime].items()}
            dow_weights = {int(k): v for k, v in profile["day_of_week_weights"][regime].items()}
            connections = _sample_timestamps(rng, month, count, hour_weights, dow_weights)

            stats = (
                spec["months"].get(month)
                or spec["quarters"].get(quarter_for(month))
                or spec["regimes"].get(regime)
                or spec["pooled"]
            )
            n_in = min(count, int(spec["monthly_request_counts"].get(month, 0)))
            has_inputs = np.zeros(count, dtype=bool)
            has_inputs[rng.choice(count, size=n_in, replace=False)] = True
            kwh_bounds = stats["bounds"]["kwh_delivered"]
            hour_bounds = stats["bounds"]["connected_hours"]

            requested = np.full(count, np.nan)
            minutes = np.full(count, np.nan)
            kwh = np.empty(count)
            hours = np.empty(count)

            block = _pick(profile, spec, month, "with_inputs")
            draws = _draw_copula(
                rng,
                block["copula_corr"],
                [
                    block["log_kwh_requested_q"],
                    block["log_minutes_available_q"],
                    block["energy_residual_q"],
                    block["dwell_residual_q"],
                ],
                probs,
                n_in,
            )
            design = np.column_stack([np.ones(n_in), draws[:, 0], draws[:, 1]])
            # Round as emitted, so the delivered/requested cap holds on the written values.
            requested[has_inputs] = np.round(np.exp(draws[:, 0]), 2)
            minutes[has_inputs] = np.exp(draws[:, 1])
            kwh[has_inputs] = np.exp(design @ np.array(block["energy_coef"]) + draws[:, 2])
            hours[has_inputs] = np.exp(design @ np.array(block["dwell_coef"]) + draws[:, 3])

            block = _pick(profile, spec, month, "without_inputs")
            draws = _draw_copula(
                rng,
                block["copula_corr"],
                [block["log_kwh_delivered_q"], block["log_connected_hours_q"]],
                probs,
                count - n_in,
            )
            kwh[~has_inputs] = np.exp(draws[:, 0])
            hours[~has_inputs] = np.exp(draws[:, 1])

            hours = np.clip(hours, *hour_bounds)
            kwh = np.clip(kwh, *kwh_bounds)
            # Physical caps: a session cannot draw more than the site's observed
            # peak rate, and requests are not exceeded grossly (data contract).
            kwh = np.minimum(kwh, hours * spec["max_kw"])
            kwh = np.where(
                has_inputs,
                np.minimum(kwh, np.nan_to_num(requested, nan=np.inf) * MAX_DELIVERED_TO_REQUESTED),
                kwh,
            )
            station_picks = rng.integers(0, len(stations), size=count)
            cluster_picks = rng.integers(0, len(clusters), size=count)
            wh_picks = rng.choice(wh_values, size=count, p=wh_p)

            for i in range(count):
                connect = connections[i].replace(tzinfo=zone).astimezone(timezone.utc)
                disconnect = connect + timedelta(hours=float(hours[i]))
                user_inputs: list[dict[str, Any]] = []
                if has_inputs[i]:
                    wh_per_mile = int(wh_picks[i])
                    user_inputs = [
                        {
                            "kWhRequested": round(float(requested[i]), 2),
                            "milesRequested": int(round(requested[i] * 1000 / wh_per_mile)),
                            "minutesAvailable": int(round(minutes[i])),
                            "WhPerMile": wh_per_mile,
                            "requestedDeparture": (
                                connect + timedelta(minutes=float(minutes[i]))
                            ).strftime(HTTP_DATE_FMT),
                            "paymentRequired": True,
                            "modifiedAt": connect.strftime(HTTP_DATE_FMT),
                        }
                    ]
                rows.append(
                    {
                        "sessionID": f"SYNTH-{site}-{month}-{i:05d}",
                        "siteID": site,
                        "clusterID": clusters[int(cluster_picks[i])],
                        "stationID": stations[int(station_picks[i])],
                        "spaceID": f"SP-{int(station_picks[i]) + 1:03d}",
                        "timezone": spec["timezone"],
                        "connectionTime": connect.strftime(HTTP_DATE_FMT),
                        "disconnectTime": disconnect.strftime(HTTP_DATE_FMT),
                        "doneChargingTime": None,
                        "kWhDelivered": round(float(kwh[i]), 3),
                        "userInputs": user_inputs,
                    }
                )

    frame = pd.DataFrame(rows)
    order = pd.to_datetime(frame["connectionTime"], format=HTTP_DATE_FMT)
    return frame.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)


def write_dataset(frame: pd.DataFrame, directory: Any, seed: int, profile: dict[str, Any]) -> dict[str, Any]:
    """Write gzipped JSONL (fixed mtime) and a manifest hashing the uncompressed bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DATA_FILE
    digest = hashlib.sha256()
    with target.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
    ) as stream:
        for record in frame.to_dict(orient="records"):
            line = (
                json.dumps(record, separators=(",", ":"), sort_keys=True, default=str) + "\n"
            ).encode("utf-8")
            stream.write(line)
            digest.update(line)

    months = pd.to_datetime(frame["connectionTime"], format=HTTP_DATE_FMT).dt.strftime("%Y-%m")
    manifest = {
        "source": "SYNTHETIC - generated from aggregate statistics, not real records",
        "generator": "evcharge.fixtures.synthetic",
        "profile": f"configs/{PROFILE_NAME}.yaml",
        "seed": seed,
        "window": profile["derived_from"]["window"],
        "regimes": profile["regimes"],
        "file": DATA_FILE,
        "sessions": len(frame),
        "sessions_by_site": {
            str(k): int(v) for k, v in frame["siteID"].value_counts().sort_index().items()
        },
        "first_month_utc": months.min(),
        "last_month_utc": months.max(),
        "uncompressed_sha256": digest.hexdigest(),
        "contains_pii": False,
        "contains_real_records": False,
        "note": (
            "Every session is drawn from aggregate parameters with a fixed seed. Results "
            "computed on this data demonstrate the pipeline and are not comparable with "
            "the licensed benchmarks."
        ),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def uncompressed_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_command() -> int:
    settings = get_settings()
    pattern = str(settings.interim_dir / "raw" / "sessions" / "**" / "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(
            "[FAIL] No normalised licensed sessions. Run:\n"
            "       python -m evcharge.ingest.normalise --source raw",
            file=sys.stderr,
        )
        return 2
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    profile = build_profile(frame)
    out = CONFIG_DIR / f"{PROFILE_NAME}.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(profile, fh, sort_keys=False, default_flow_style=None, width=100)
    print(f"[ok]   profiled {profile['derived_from']['sessions_profiled']:,} sessions")
    print(f"[ok]   profile -> {out}")
    return 0


def _generate_command(seed: int) -> int:
    settings = get_settings()
    profile = load_config(PROFILE_NAME)
    frame = generate(profile, seed)
    manifest = write_dataset(frame, settings.source_dir("synthetic"), seed, profile)
    print(f"[ok]   generated {manifest['sessions']:,} synthetic sessions")
    print(f"[ok]   sites: {manifest['sessions_by_site']}")
    print(f"[ok]   sha256 (uncompressed): {manifest['uncompressed_sha256'][:16]}...")
    print(f"[ok]   data -> {settings.source_dir('synthetic') / DATA_FILE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("generate", "profile"), nargs="?", default="generate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        if args.command == "profile":
            return _profile_command()
        return _generate_command(args.seed)
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
