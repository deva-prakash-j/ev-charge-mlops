"""The extract manifest is how an evaluator verifies the data they were given."""

import hashlib
import json

import pytest

from evcharge.config import get_settings
from evcharge.ingest.fetch_sessions import fetch_site


class StubClient:
    def __init__(self, sessions):
        self._sessions = sessions

    def iter_sessions(self, site):
        yield from self._sessions


def test_manifest_hash_is_the_file_checksum(tmp_path):
    sessions = [
        {"sessionID": "a", "kWhDelivered": 1.5, "siteID": 2},
        {"sessionID": "b", "kWhDelivered": 7.25, "siteID": "0002"},
    ]

    entry = fetch_site(StubClient(sessions), "caltech", tmp_path)

    written = (tmp_path / entry["file"]).read_bytes()
    assert entry["sha256"] == hashlib.sha256(written).hexdigest()
    assert entry["bytes"] == len(written)
    assert entry["sessions"] == 2
    assert b"\r" not in written


def test_local_extract_matches_its_manifest():
    """Runs only where the real extract is present, such as the upGrad submission."""
    raw_dir = get_settings().raw_dir
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no extract manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sites = manifest["sites"]
    if not all((raw_dir / site["file"]).is_file() for site in sites):
        pytest.skip("licensed extract not present")

    for site in sites:
        data = (raw_dir / site["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == site["sha256"], site["file"]
        assert len(data) == site["bytes"], site["file"]
        assert data.count(b"\n") == site["sessions"], site["file"]
    assert manifest["total_sessions"] == sum(site["sessions"] for site in sites)
