import hashlib
import json

import pandas as pd
import pytest

from evcharge.config import ConfigError, Settings, get_settings
from evcharge.validation.reference import (
    SNAPSHOT_COLUMNS,
    build_reference_frame,
    create_reference_snapshot,
    write_reference_snapshot,
)


def _source_frame():
    return pd.DataFrame(
        {
            "sessionID": ["old", "in-window", "future"],
            "userID": ["private-1", "private-2", "private-3"],
            "connectionTime": pd.to_datetime(
                ["2019-06-30T23:00:00Z", "2019-07-01T00:00:00Z", "2020-03-01T00:00:00Z"],
                utc=True,
            ),
            "requestedDeparture": pd.to_datetime(
                ["2019-07-01T00:00:00Z", "2019-07-01T01:30:00Z", "2020-03-01T02:00:00Z"],
                utc=True,
            ),
            "connectionTimeLocal": pd.to_datetime(
                ["2019-06-30", "2019-07-01", "2020-03-01"]
            ),
            "siteID": ["0001", "0001", "0002"],
            "stationID": ["0001-S000", "0001-S001", "0002-S000"],
            "clusterID": ["0001", "0001", "0002"],
            "kWhRequested": [10.0, 20.0, 30.0],
            "milesRequested": [20.0, 40.0, 60.0],
            "minutesAvailable": [60.0, 120.0, 180.0],
            "WhPerMile": [300.0, 300.0, 300.0],
            "kWhDelivered": [8.0, 15.0, 25.0],
            "connectedHours": [1.0, 2.0, 3.0],
        }
    )


def test_reference_snapshot_is_half_open_and_excludes_identifiers(tmp_path):
    data_config = {
        "split": {
            "reference_window_start": "2019-07-01",
            "reference_window_end_exclusive": "2020-03-01",
        }
    }
    snapshot = build_reference_frame(
        _source_frame(), data_config, {"quarantine_rules": []}
    )

    assert len(snapshot) == 1
    assert snapshot.columns.tolist() == list(SNAPSHOT_COLUMNS)
    assert "sessionID" not in snapshot and "userID" not in snapshot
    assert snapshot["declared_delivered_gap_kwh"].iat[0] == -5.0
    assert snapshot["declared_duration_minutes"].iat[0] == 90.0


def test_reference_manifest_checksums_file_and_counts_rows(tmp_path):
    frame = build_reference_frame(
        _source_frame(),
        {
            "split": {
                "reference_window_start": "2019-07-01",
                "reference_window_end_exclusive": "2020-03-01",
            }
        },
        {"quarantine_rules": []},
    )
    manifest = write_reference_snapshot(
        frame, tmp_path, "source-hash", "2019-07-01", "2020-03-01"
    )

    snapshot_path = tmp_path / manifest["file"]
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["rows"] == 1
    assert manifest["rows_by_month"] == {"2019-07": 1}
    assert manifest["rows_by_site"] == {"0001": 1}
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_synthetic_reference_is_frozen_beside_the_synthetic_source(tmp_path):
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")
    source_manifest = settings.source_dir("synthetic") / "manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text('{"sessions": 3}\n', encoding="utf-8")
    partition = (
        settings.interim_dir / "synthetic" / "sessions" / "site=0001" / "month=2019-07"
    )
    partition.mkdir(parents=True)
    _source_frame().to_parquet(partition / "sessions.parquet", index=False)

    manifest = create_reference_snapshot(settings, "synthetic")

    reference_dir = tmp_path / "synthetic" / "reference"
    assert json.loads((reference_dir / "manifest.json").read_text()) == manifest
    assert (reference_dir / manifest["file"]).is_file()
    assert manifest["rows"] == 1
    assert (
        manifest["source_extract_manifest_sha256"]
        == hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    )
    assert not (tmp_path / "reference").exists()


def test_sample_source_cannot_build_a_drift_reference(tmp_path):
    with pytest.raises(ConfigError, match="drift reference"):
        create_reference_snapshot(Settings(data_dir=tmp_path), "sample")


def test_local_frozen_snapshot_matches_its_manifest():
    reference_dir = get_settings().data_dir / "reference"
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("reference snapshot has not been generated from the licensed extract")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_path = reference_dir / manifest["file"]
    if not snapshot_path.is_file():
        pytest.skip("licensed reference snapshot is not present in this copy")

    frame = pd.read_parquet(snapshot_path)
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == manifest["sha256"]
    assert len(frame) == manifest["rows"]
    assert frame.columns.tolist() == manifest["columns"]
    assert (
        hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=False).values.tobytes()
        ).hexdigest()
        == manifest["logical_sha256"]
    )
    assert sum(manifest["rows_by_month"].values()) == len(frame)
    assert sum(manifest["rows_by_site"].values()) == len(frame)
