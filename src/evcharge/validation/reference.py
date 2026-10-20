"""Create a versioned, de-identified drift reference snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from evcharge.config import ConfigError, Settings, get_settings, load_config
from evcharge.validation.contract import apply_quarantine, load_normalised

SNAPSHOT_VERSION = "v1"
SNAPSHOT_COLUMNS = (
    "connectionTimeLocal",
    "siteID",
    "stationID",
    "clusterID",
    "kWhRequested",
    "milesRequested",
    "minutesAvailable",
    "WhPerMile",
    "declared_duration_minutes",
    "kWhDelivered",
    "connectedHours",
    "declared_delivered_gap_kwh",
)


def build_reference_frame(
    frame: pd.DataFrame, data_config: dict[str, Any], contract_config: dict[str, Any]
) -> pd.DataFrame:
    """Select the frozen interval and monitoring fields, excluding record IDs."""
    clean, _ = apply_quarantine(frame, contract_config["quarantine_rules"])
    window_start = pd.Timestamp(data_config["split"]["reference_window_start"])
    window_end = pd.Timestamp(data_config["split"]["reference_window_end_exclusive"])
    times = pd.to_datetime(clean["connectionTimeLocal"])
    selected = clean.loc[
        (times >= window_start)
        & (times < window_end)
        & clean["kWhDelivered"].notna()
    ].copy()
    if "connectionTime" not in selected or "requestedDeparture" not in selected:
        raise ConfigError(
            "Reference input must include connectionTime and requestedDeparture "
            "to derive the plug-in-time duration feature."
        )
    selected["declared_duration_minutes"] = (
        pd.to_datetime(selected["requestedDeparture"], utc=True, errors="coerce")
        - pd.to_datetime(selected["connectionTime"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    selected["declared_delivered_gap_kwh"] = (
        pd.to_numeric(selected["kWhDelivered"], errors="coerce")
        - pd.to_numeric(selected["kWhRequested"], errors="coerce")
    )
    missing = sorted(set(SNAPSHOT_COLUMNS) - set(selected.columns))
    if missing:
        raise ConfigError(f"Reference snapshot fields missing: {missing}")
    snapshot = selected.loc[:, SNAPSHOT_COLUMNS]
    return snapshot.sort_values(
        list(SNAPSHOT_COLUMNS), kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reference_snapshot(
    frame: pd.DataFrame,
    output_dir: Path,
    source_manifest_sha256: str,
    window_start: str,
    window_end_exclusive: str,
) -> dict[str, Any]:
    """Persist the snapshot and a data-free manifest describing its contents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"drift_reference_{SNAPSHOT_VERSION}.parquet"
    frame.to_parquet(snapshot_path, index=False, compression="zstd")
    manifest = {
        "snapshot_version": SNAPSHOT_VERSION,
        "file": snapshot_path.name,
        "sha256": _sha256(snapshot_path),
        "logical_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=False).values.tobytes()
        ).hexdigest(),
        "rows": len(frame),
        "columns": list(frame.columns),
        "window": {
            "start_inclusive": window_start,
            "end_exclusive": window_end_exclusive,
        },
        "rows_by_month": {
            str(month): int(count)
            for month, count in frame["connectionTimeLocal"]
            .dt.to_period("M")
            .value_counts()
            .sort_index()
            .items()
        },
        "rows_by_site": {
            str(site): int(count)
            for site, count in frame["siteID"].value_counts().sort_index().items()
        },
        "source_extract_manifest_sha256": source_manifest_sha256,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def create_reference_snapshot(settings: Settings, source: str = "raw") -> dict[str, Any]:
    if source != "raw":
        raise ConfigError("The versioned drift reference must be built from source 'raw'.")
    raw_manifest = settings.raw_dir / "manifest.json"
    if not raw_manifest.is_file():
        raise ConfigError(f"Raw extract manifest is missing: {raw_manifest}")
    contract = load_config("contract")
    data = load_config("data")
    source_manifest_sha256 = _sha256(raw_manifest)
    frame = build_reference_frame(load_normalised(settings, source), data, contract)
    return write_reference_snapshot(
        frame,
        settings.data_dir / "reference",
        source_manifest_sha256,
        data["split"]["reference_window_start"],
        data["split"]["reference_window_end_exclusive"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the drift reference snapshot.")
    parser.add_argument("--source", choices=("raw",), default="raw")
    args = parser.parse_args(argv)
    try:
        manifest = create_reference_snapshot(get_settings(), args.source)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(
        f"[ok]   {manifest['rows']:,} reference rows "
        f"({manifest['window']['start_inclusive']}.."
        f"{manifest['window']['end_exclusive']} exclusive)"
    )
    print(f"[ok]   snapshot SHA-256: {manifest['sha256']}")
    print("[ok]   manifest -> data/reference/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
