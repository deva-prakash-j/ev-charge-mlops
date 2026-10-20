"""The synthetic dataset is published, so these tests guard what it may contain."""

import copy
import gzip
import json
import re

import numpy as np
import pandas as pd
import pytest

from evcharge.config import ConfigError, Settings, get_settings, load_config
from evcharge.fixtures.generate import HTTP_DATE_FMT
from evcharge.fixtures.synthetic import (
    DATA_FILE,
    DEFAULT_SEED,
    MAX_DELIVERED_TO_REQUESTED,
    MIN_OBSERVATIONS,
    PROFILE_NAME,
    build_profile,
    generate,
    quarter_for,
    regime_for,
    uncompressed_sha256,
    write_dataset,
)
from evcharge.ingest.normalise import normalise_site

FORBIDDEN_FIELDS = {"userID", "_id", "user_id"}


@pytest.fixture(scope="module")
def profile():
    return load_config(PROFILE_NAME)


@pytest.fixture(scope="module")
def small_profile(profile):
    """Same parameters at 5% volume, so generation tests stay fast."""
    reduced = copy.deepcopy(profile)
    for spec in reduced["sites"].values():
        spec["monthly_counts"] = {
            month: max(1, count // 20) for month, count in spec["monthly_counts"].items()
        }
        spec["monthly_request_counts"] = {
            month: min(spec["monthly_counts"][month], max(1, count // 20))
            for month, count in spec["monthly_request_counts"].items()
        }
    return reduced


@pytest.fixture(scope="module")
def small_frame(small_profile):
    return generate(small_profile, seed=DEFAULT_SEED)


def _fitted_blocks(profile):
    yield profile["all_sites_by_regime"].values()
    for spec in profile["sites"].values():
        yield [spec["pooled"]]
        yield spec["regimes"].values()
        yield spec["quarters"].values()
        yield spec["months"].values()


def test_profile_blocks_are_aggregates_of_enough_sessions(profile):
    probs = profile["quantile_probs"]
    assert 0 < probs[0] and probs[-1] < 1, "extreme order statistics would describe one session"
    for group in _fitted_blocks(profile):
        for stats in group:
            assert stats["n"] >= MIN_OBSERVATIONS
            for name in ("with_inputs", "without_inputs"):
                block = stats[name]
                if block is not None:
                    assert block["n"] >= MIN_OBSERVATIONS


def test_profile_contains_no_identifier_fields(profile):
    text = json.dumps(profile)
    assert FORBIDDEN_FIELDS.isdisjoint(re.findall(r'"(\w+)":', text))


def test_profile_covers_the_full_licensed_period(profile):
    months = sorted(
        month for spec in profile["sites"].values() for month in spec["monthly_counts"]
    )
    assert months[0] == "2018-04" and months[-1] == "2021-09"
    assert profile["derived_from"]["sessions_profiled"] == sum(
        sum(spec["monthly_counts"].values()) for spec in profile["sites"].values()
    )


def test_generation_is_deterministic(small_profile, small_frame):
    assert small_frame.equals(generate(small_profile, seed=DEFAULT_SEED))
    assert not small_frame.equals(generate(small_profile, seed=DEFAULT_SEED + 1))


def test_generated_sessions_are_synthetic_and_identifier_free(small_frame):
    assert FORBIDDEN_FIELDS.isdisjoint(small_frame.columns)
    assert small_frame["sessionID"].str.startswith("SYNTH-").all()
    assert small_frame["sessionID"].is_unique
    for inputs in small_frame["userInputs"]:
        for entry in inputs:
            assert FORBIDDEN_FIELDS.isdisjoint(entry)


def test_generated_sessions_satisfy_the_data_contract_domains(small_frame):
    contract = load_config("contract")
    for site, pattern in contract["station_id_domains"].items():
        stations = small_frame.loc[small_frame["siteID"] == site, "stationID"]
        assert stations.str.fullmatch(pattern).all()
    assert small_frame["kWhDelivered"].between(0, 200, inclusive="neither").all()
    connect = pd.to_datetime(small_frame["connectionTime"], format=HTTP_DATE_FMT)
    disconnect = pd.to_datetime(small_frame["disconnectTime"], format=HTTP_DATE_FMT)
    hours = (disconnect - connect).dt.total_seconds() / 3600
    assert hours.between(0, 168, inclusive="right").all()
    for kwh, inputs in zip(small_frame["kWhDelivered"], small_frame["userInputs"]):
        if inputs:
            assert kwh <= inputs[0]["kWhRequested"] * MAX_DELIVERED_TO_REQUESTED + 0.01


def test_generation_preserves_the_2020_collapse(profile):
    def total(month):
        return sum(spec["monthly_counts"].get(month, 0) for spec in profile["sites"].values())

    assert total("2020-04") < 0.3 * total("2020-02"), "H1 needs a >=70% volume fall"


def test_every_month_keeps_its_profiled_request_count(small_profile, small_frame):
    """The contract blocks a batch with no driver requests, so none may be dropped."""
    months = pd.to_datetime(small_frame["connectionTime"], format=HTTP_DATE_FMT, utc=True)
    local = months.dt.tz_convert("America/Los_Angeles").dt.strftime("%Y-%m")
    with_inputs = small_frame["userInputs"].map(bool)
    observed = with_inputs.groupby([small_frame["siteID"], local]).sum()
    for site, spec in small_profile["sites"].items():
        for month, expected in spec["monthly_request_counts"].items():
            assert observed.get((site, month), 0) == expected
    for spec in small_profile["sites"].values():
        assert set(spec["monthly_request_counts"]) <= set(spec["monthly_counts"])
    batch_requests = with_inputs.groupby(local).sum()
    assert (batch_requests > 0).all()


def test_period_lookup_helpers():
    assert regime_for("2020-02") == "pre_shift"
    assert regime_for("2020-04") == "shutdown"
    assert regime_for("2021-09") == "reopening"
    assert quarter_for("2020-04") == "2020Q2/shutdown"
    assert quarter_for("2020-02") == "2020Q1/pre_shift"
    assert quarter_for("2020-03") == "2020Q1/shutdown"
    with pytest.raises(ValueError):
        regime_for("2017-12")


def test_build_profile_drops_blocks_below_the_session_floor():
    rng = np.random.default_rng(0)
    rows = 400
    months = np.where(np.arange(rows) < 380, "2019-08", "2020-04")
    requested = np.where(rng.random(rows) < 0.8, rng.uniform(5, 40, rows), np.nan)
    frame = pd.DataFrame(
        {
            "siteID": "0001",
            "stationID": "1-1-1-1",
            "clusterID": "0001",
            "timezone": "America/Los_Angeles",
            "connectionTimeLocal": pd.to_datetime([f"{m}-10 08:00" for m in months]),
            "kWhDelivered": rng.uniform(1, 30, rows),
            "connectedHours": rng.uniform(1, 10, rows),
            "kWhRequested": requested,
            "minutesAvailable": np.where(np.isnan(requested), np.nan, rng.uniform(60, 600, rows)),
            "WhPerMile": 300,
        }
    )
    profile = build_profile(frame)
    site = profile["sites"]["0001"]

    assert site["monthly_counts"] == {"2019-08": 380, "2020-04": 20}
    assert site["monthly_request_counts"] == {
        "2019-08": int((requested[:380] > 0).sum()),
        "2020-04": int((requested[380:] > 0).sum()),
    }
    assert set(site["months"]) == {"2019-08"}
    assert set(site["quarters"]) == {"2019Q3/pre_shift"}
    assert set(site["regimes"]) == {"pre_shift"}
    assert site["quarters"]["2019Q3/pre_shift"]["with_inputs"]["n"] >= MIN_OBSERVATIONS
    assert site["quarters"]["2019Q3/pre_shift"]["without_inputs"]["n"] >= MIN_OBSERVATIONS
    assert len(site["pooled"]["with_inputs"]["copula_corr"]) == 4


def test_dataset_round_trips_through_gzip_and_normalise(small_profile, small_frame, tmp_path):
    manifest = write_dataset(small_frame, tmp_path, DEFAULT_SEED, small_profile)
    path = tmp_path / DATA_FILE
    assert uncompressed_sha256(path) == manifest["uncompressed_sha256"]
    assert manifest["contains_real_records"] is False

    normalised = normalise_site(path, load_config("data")["timestamps"]["format"])
    assert len(normalised) == len(small_frame)
    assert normalised["connectedHours"].gt(0).all()
    assert normalised["year_month"].min() == "2018-04"


def test_committed_dataset_matches_its_manifest():
    directory = get_settings().source_dir("synthetic")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_real_records"] is False
    assert manifest["contains_pii"] is False

    path = directory / manifest["file"]
    assert uncompressed_sha256(path) == manifest["uncompressed_sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == manifest["sessions"]
    assert all(FORBIDDEN_FIELDS.isdisjoint(record) for record in records)
    assert all(record["sessionID"].startswith("SYNTH-") for record in records)


def test_source_directories_keep_the_licensed_reference_path(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.source_dir("synthetic") == tmp_path / "synthetic"
    assert settings.reference_dir("raw") == tmp_path / "reference"
    assert settings.reference_dir("synthetic") == tmp_path / "synthetic" / "reference"
    with pytest.raises(ConfigError):
        settings.reference_dir("sample")
    with pytest.raises(ConfigError):
        settings.source_dir("other")
