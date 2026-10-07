"""The fixture is published, so these tests guard what it may contain."""

import json

import pandas as pd
import pytest

from evcharge.config import get_settings, load_config
from evcharge.fixtures.generate import DEFAULT_SEED, HTTP_DATE_FMT, generate
from evcharge.ingest.normalise import earliest_user_input, normalise_site_id

FORBIDDEN_FIELDS = {"userID", "_id", "user_id"}


@pytest.fixture(scope="module")
def profile():
    return load_config("sample_profile")


def test_fixture_contains_no_identifiers(profile):
    frame = generate(profile, seed=DEFAULT_SEED)
    assert FORBIDDEN_FIELDS.isdisjoint(frame.columns)
    for inputs in frame["userInputs"]:
        for entry in inputs:
            assert FORBIDDEN_FIELDS.isdisjoint(entry)


def test_fixture_session_ids_are_synthetic(profile):
    frame = generate(profile, seed=DEFAULT_SEED)
    assert frame["sessionID"].str.startswith("SYN-").all()


def test_generation_is_deterministic(profile):
    first = generate(profile, seed=DEFAULT_SEED)
    second = generate(profile, seed=DEFAULT_SEED)
    assert first.equals(second)


def test_different_seed_changes_output(profile):
    assert not generate(profile, seed=1).equals(generate(profile, seed=2))


def _local_months(frame, profile):
    """Month labels in site-local time, matching how partitions are keyed."""
    times = pd.to_datetime(frame["connectionTime"], format=HTTP_DATE_FMT, utc=True)
    tz = next(iter(profile["sites"].values()))["timezone"]
    return times.dt.tz_convert(tz).dt.strftime("%Y-%m")


def test_fixture_preserves_the_2020_volume_collapse(profile):
    frame = generate(profile, seed=DEFAULT_SEED)
    months = _local_months(frame, profile)
    stable = (months == "2020-02").sum()
    collapsed = (months == "2020-04").sum()
    assert stable > 0
    assert collapsed < stable * 0.5, "the regime shift must survive downsampling"


def test_fixture_keeps_the_local_workday_arrival_peak(profile):
    """Arrival hour is a core feature; it must not be shifted by timezone handling."""
    frame = generate(profile, seed=DEFAULT_SEED)
    times = pd.to_datetime(frame["connectionTime"], format=HTTP_DATE_FMT, utc=True)
    tz = next(iter(profile["sites"].values()))["timezone"]
    hours = times.dt.tz_convert(tz).dt.hour
    morning = hours.between(5, 10).mean()
    overnight = hours.between(0, 4).mean()
    assert morning > 0.4, f"expected a morning commute peak, got {morning:.2f}"
    assert overnight < 0.1, f"unexpected overnight mass: {overnight:.2f}"


def test_committed_fixture_matches_its_manifest():
    settings = get_settings()
    manifest_path = settings.sample_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("fixture not generated")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["contains_real_records"] is False
    assert manifest["contains_pii"] is False

    lines = (settings.sample_dir / "sessions_sample.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == manifest["sessions"]
    for line in lines:
        assert FORBIDDEN_FIELDS.isdisjoint(json.loads(line))


@pytest.mark.parametrize(
    "raw, expected",
    [(2, "0002"), ("0002", "0002"), ("0001", "0001"), (19, "0019"), (None, "unknown")],
)
def test_mixed_type_site_ids_collapse(raw, expected):
    """The feed emits siteID as both int 2 and str '0002'."""
    assert normalise_site_id(raw) == expected


def test_only_the_earliest_user_input_is_kept():
    """Later revisions are not known at plug-in time and would leak the future."""
    revisions = [
        {"modifiedAt": "Wed, 05 Sep 2018 12:14:30 GMT", "kWhRequested": 30.0},
        {"modifiedAt": "Wed, 05 Sep 2018 11:08:31 GMT", "kWhRequested": 8.0},
        {"modifiedAt": "Wed, 05 Sep 2018 11:25:51 GMT", "kWhRequested": 20.0},
    ]
    result = earliest_user_input(revisions)
    assert result["kWhRequested"] == 8.0
    assert result["userInputs_count"] == 3


def test_missing_user_inputs_are_handled():
    assert earliest_user_input([])["userInputs_count"] == 0
    assert earliest_user_input(None)["kWhRequested"] is None
