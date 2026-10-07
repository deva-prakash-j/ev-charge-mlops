"""Train/serve parity and leakage are the two failure modes that silently
destroy a deployed model, so both are asserted here rather than reasoned about.
"""

import glob

import numpy as np
import pandas as pd
import pytest

from evcharge.config import get_settings, load_config
from evcharge.features.build import (
    FeatureSpec,
    LeakageError,
    assert_no_leakage,
    build_xy,
    fit,
    transform,
)


@pytest.fixture(scope="module")
def configs():
    return load_config("features"), load_config("data")


@pytest.fixture(scope="module")
def sessions():
    """Real normalised rows from the synthetic fixture, not a hand-built frame."""
    settings = get_settings()
    files = glob.glob(
        str(settings.interim_dir / "sample" / "sessions" / "**" / "*.parquet"),
        recursive=True,
    )
    if not files:
        pytest.skip("run: python -m evcharge.ingest.normalise --source sample")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frame.sort_values("connectionTimeLocal").reset_index(drop=True)


@pytest.fixture(scope="module")
def spec(sessions, configs):
    feature_config, data_config = configs
    return fit(sessions, feature_config, data_config)


def test_batch_and_single_row_transforms_are_identical(sessions, spec, configs):
    """The serving path scores one row at a time; it must match training exactly."""
    feature_config, data_config = configs
    sample = sessions.head(60)

    batch = transform(sample, spec, feature_config, data_config)
    row_by_row = pd.concat(
        [
            transform(sample.iloc[[i]], spec, feature_config, data_config)
            for i in range(len(sample))
        ]
    )

    pd.testing.assert_frame_equal(batch, row_by_row, check_exact=False, rtol=1e-12)


def test_row_order_does_not_change_features(sessions, spec, configs):
    """A feature that depended on batch statistics would fail this."""
    feature_config, data_config = configs
    sample = sessions.head(50)

    straight = transform(sample, spec, feature_config, data_config)
    shuffled = transform(
        sample.iloc[::-1], spec, feature_config, data_config
    ).iloc[::-1]

    pd.testing.assert_frame_equal(straight, shuffled, check_exact=False, rtol=1e-12)


def test_post_hoc_columns_are_rejected(configs):
    _, data_config = configs
    for column in ("kWhDelivered", "disconnectTime", "doneChargingTime"):
        with pytest.raises(LeakageError, match="plug-in time"):
            assert_no_leakage(["arrival_hour", column], data_config)


def test_identifier_columns_are_rejected(configs):
    _, data_config = configs
    with pytest.raises(LeakageError):
        assert_no_leakage(["arrival_hour", "userID"], data_config)


def test_feature_matrix_excludes_the_target(sessions, spec, configs):
    feature_config, data_config = configs
    features, target = build_xy(sessions.head(20), spec, feature_config, data_config)
    assert feature_config["target"] not in features.columns
    assert len(features) == len(target)


def test_unknown_category_maps_to_sentinel(sessions, spec, configs):
    """Serving will see stations that did not exist during training."""
    feature_config, data_config = configs
    row = sessions.head(1).copy()
    row["stationID"] = "STATION-THAT-DID-NOT-EXIST"
    features = transform(row, spec, feature_config, data_config)
    assert features["stationID_code"].iat[0] == spec.unknown_category_code


def test_hour_encoding_is_cyclical(sessions, spec, configs):
    """23:00 and 00:00 must be adjacent, not maximally distant."""
    feature_config, data_config = configs
    rows = sessions.head(2).copy()
    rows["connectionTimeLocal"] = pd.to_datetime(
        ["2020-01-01 23:00:00", "2020-01-02 00:00:00"]
    )
    out = transform(rows, spec, feature_config, data_config)

    def distance(a, b):
        return np.hypot(
            out["arrival_hour_sin"].iat[a] - out["arrival_hour_sin"].iat[b],
            out["arrival_hour_cos"].iat[a] - out["arrival_hour_cos"].iat[b],
        )

    assert distance(0, 1) < 0.3


def test_missing_user_inputs_are_flagged_not_silently_zero(sessions, spec, configs):
    feature_config, data_config = configs
    row = sessions.head(1).copy()
    row["kWhRequested"] = np.nan
    features = transform(row, spec, feature_config, data_config)
    assert features["has_user_inputs"].iat[0] == 0.0
    assert pd.isna(features["kWhRequested"].iat[0])


def test_zero_valued_requests_do_not_produce_infinities(sessions, spec, configs):
    feature_config, data_config = configs
    row = sessions.head(1).copy()
    row["minutesAvailable"] = 0
    row["milesRequested"] = 0
    features = transform(row, spec, feature_config, data_config)
    assert np.isfinite(features.to_numpy(dtype="float64")).all() or features.isna().any().any()
    assert not np.isinf(features.to_numpy(dtype="float64")).any()


def test_column_order_is_stable(sessions, spec, configs):
    feature_config, data_config = configs
    first = transform(sessions.head(10), spec, feature_config, data_config)
    second = transform(sessions.tail(10), spec, feature_config, data_config)
    assert first.columns.tolist() == second.columns.tolist() == spec.feature_names


def test_spec_round_trips_through_disk(spec, tmp_path):
    path = tmp_path / "feature_spec.json"
    spec.save(path)
    restored = FeatureSpec.load(path)
    assert restored.categories == spec.categories
    assert restored.feature_names == spec.feature_names
