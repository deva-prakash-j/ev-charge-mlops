"""Normalise raw ACN-Data JSONL into a flat, monthly-partitioned Parquet table.

Normalisation only — no modelling features. Two correctness rules are enforced
here because getting them wrong silently invents a model that cannot be served:

1. `userInputs` is a list of driver revisions. Only the earliest entry exists at
   plug-in time, so later revisions are dropped.
2. Month partitions use each site's local timezone, not GMT, so "March 2020"
   means the operator's March.

Usage:  python -m evcharge.ingest.normalise
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from evcharge.config import ConfigError, Settings, get_settings, load_config

logger = logging.getLogger(__name__)

SOURCES = ("sample", "raw")

# Columns the feed types inconsistently; coerced so Parquet gets a stable schema.
IDENTIFIER_COLUMNS = ("siteID", "clusterID", "stationID", "spaceID", "userID")

USER_INPUT_FIELDS = (
    "kWhRequested",
    "milesRequested",
    "minutesAvailable",
    "WhPerMile",
    "requestedDeparture",
    "paymentRequired",
)


def earliest_user_input(raw: Any) -> dict[str, Any]:
    """Return the first driver request; later revisions are not known at plug-in."""
    if not isinstance(raw, list) or not raw:
        return {field: None for field in USER_INPUT_FIELDS} | {"userInputs_count": 0}

    def sort_key(entry: dict[str, Any]) -> str:
        return str(entry.get("modifiedAt") or "")

    first = min(raw, key=sort_key)
    flat = {field: first.get(field) for field in USER_INPUT_FIELDS}
    flat["userInputs_count"] = len(raw)
    return flat


def flatten_sessions(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            session = json.loads(line)
            flat = {
                key: value
                for key, value in session.items()
                if key != "userInputs"
            }
            flat.update(earliest_user_input(session.get("userInputs")))
            records.append(flat)
    return pd.DataFrame.from_records(records)


def parse_times(frame: pd.DataFrame, fmt: str) -> pd.DataFrame:
    for column in ("connectionTime", "disconnectTime", "doneChargingTime",
                   "requestedDeparture"):
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column], format=fmt, utc=True, errors="coerce"
            )
    return frame


def add_local_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive local connection time and the year-month partition key."""
    local = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for tz, group in frame.groupby("timezone", dropna=True):
        converted = group["connectionTime"].dt.tz_convert(tz).dt.tz_localize(None)
        local.loc[group.index] = converted

    frame["connectionTimeLocal"] = local
    frame["year_month"] = local.dt.strftime("%Y-%m")
    return frame


def normalise_site(path: Path, fmt: str) -> pd.DataFrame:
    frame = flatten_sessions(path)
    for column in IDENTIFIER_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].map(normalise_identifier)
    frame = parse_times(frame, fmt)
    frame = add_local_partition(frame)

    frame["connectedHours"] = (
        frame["disconnectTime"] - frame["connectionTime"]
    ).dt.total_seconds() / 3600.0

    unparsed = int(frame["connectionTime"].isna().sum())
    if unparsed:
        logger.warning("%s: %d rows have an unparseable connectionTime", path.name, unparsed)
    return frame


def write_partitions(frame: pd.DataFrame, out_dir: Path) -> int:
    """Write to a staging tree then swap, so a failure never leaves a partial one."""
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    written = 0
    for (site, month), group in frame.groupby(["siteID", "year_month"], dropna=True):
        target = staging / f"site={site}" / f"month={month}"
        target.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["year_month"]).to_parquet(
            target / "sessions.parquet", index=False
        )
        written += 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return written


def normalise_identifier(value: Any) -> str:
    """The feed emits siteID and clusterID as both int 2 and str '0002'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{int(value):04d}"
    text = str(value).strip()
    return f"{int(text):04d}" if text.isdigit() else text


# Retained for readability at call sites that only deal with siteID.
normalise_site_id = normalise_identifier


def resolve_source(settings: Settings, source: str) -> tuple[Path, list[Path]]:
    directory = settings.sample_dir if source == "sample" else settings.raw_dir
    return directory, sorted(directory.glob("sessions*.jsonl"))


def run(settings: Settings, config: dict[str, Any], source: str = "sample") -> pd.DataFrame:
    fmt = config["timestamps"]["format"]
    directory, files = resolve_source(settings, source)
    if not files:
        hint = (
            "python -m evcharge.fixtures.generate"
            if source == "sample"
            else "python -m evcharge.ingest.fetch_sessions"
        )
        raise ConfigError(f"No '{source}' extracts in {directory}. Run: {hint}")

    frames = []
    for path in files:
        frame = normalise_site(path, fmt)
        logger.info("%s -> %d rows, %d months", path.name, len(frame),
                    frame["year_month"].nunique())
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    out_dir = settings.interim_dir / source / "sessions"
    partitions = write_partitions(combined, out_dir)
    logger.info("wrote %d site/month partitions -> %s", partitions, out_dir)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalise session extracts.")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="sample",
        help="'sample' uses the committed synthetic fixture (no credentials needed).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        settings = get_settings()
        config = load_config("data")
        frame = run(settings, config, args.source)
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(f"\n[ok]   source:  {args.source}")
    print(f"[ok]   {len(frame):,} sessions normalised")
    print(f"[ok]   columns: {len(frame.columns)}")
    print(f"[ok]   sites:   {sorted(frame['siteID'].unique())}")
    print(f"[ok]   months:  {frame['year_month'].min()} -> {frame['year_month'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
