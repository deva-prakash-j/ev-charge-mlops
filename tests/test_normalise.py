from pathlib import Path

import pandas as pd

from evcharge.ingest.normalise import write_partitions


def test_partition_replacement_restores_previous_tree_if_install_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "sessions"
    output.mkdir()
    marker = output / "previous.txt"
    marker.write_text("previous dataset", encoding="utf-8")

    original_rename = Path.rename

    def fail_staging_install(path, target):
        if path.name.endswith(".staging"):
            raise OSError("simulated install failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_install)

    frame = pd.DataFrame(
        {"siteID": ["0001"], "year_month": ["2020-01"], "value": [1]}
    )
    try:
        write_partitions(frame, output)
    except OSError as exc:
        assert "simulated install failure" in str(exc)
    else:
        raise AssertionError("expected staging install to fail")

    assert marker.read_text(encoding="utf-8") == "previous dataset"
    assert not list(tmp_path.glob("sessions.backup-*"))


def test_partition_replacement_recovers_backup_left_by_interrupted_run(
    tmp_path, monkeypatch
):
    backup = tmp_path / "sessions.backup-interrupted"
    backup.mkdir()
    marker = backup / "previous.txt"
    marker.write_text("recovered dataset", encoding="utf-8")

    original_rename = Path.rename

    def fail_staging_install(path, target):
        if path.name.endswith(".staging"):
            raise OSError("simulated install failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_install)

    frame = pd.DataFrame(
        {"siteID": ["0001"], "year_month": ["2020-01"], "value": [1]}
    )
    try:
        write_partitions(frame, tmp_path / "sessions")
    except OSError as exc:
        assert "simulated install failure" in str(exc)
    else:
        raise AssertionError("expected staging install to fail")

    assert (tmp_path / "sessions" / "previous.txt").read_text(
        encoding="utf-8"
    ) == "recovered dataset"
