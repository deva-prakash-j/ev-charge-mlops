"""The contract must block on schema breaches and quarantine implausible rows,
so both behaviours are pinned here rather than only observed in a run."""

import pandas as pd
import pytest

from evcharge.config import ConfigError, load_config
from evcharge.validation.contract import ContractBreach, apply_quarantine, validate


@pytest.fixture(scope="module")
def spec():
    return load_config("contract")


def _frame(**overrides) -> pd.DataFrame:
    base = {
        "sessionID": ["a", "b", "c"],
        "siteID": ["0001", "0001", "0002"],
        "clusterID": ["0001", "0001", "0002"],
        "stationID": ["0001-S000", "0001-S001", "0002-S000"],
        "connectionTime": pd.to_datetime(
            ["2020-01-01 08:00", "2020-01-02 08:00", "2020-01-03 08:00"], utc=True
        ),
        "disconnectTime": pd.to_datetime(
            ["2020-01-01 12:00", "2020-01-02 12:00", "2020-01-03 12:00"], utc=True
        ),
        "doneChargingTime": pd.to_datetime(
            ["2020-01-01 10:00", "2020-01-02 10:00", "2020-01-03 10:00"], utc=True
        ),
        "kWhRequested": [12.0, 12.0, 8.0],
        "kWhDelivered": [10.0, 12.0, 8.0],
        "timezone": ["America/Los_Angeles"] * 3,
        "connectedHours": [4.0, 5.0, 6.0],
        "connectionTimeLocal": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"]
        ),
        "minutesAvailable": [300, 400, 500],
        "userInputs_count": [1, 1, 1],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_week_long_sessions_are_quarantined(spec):
    frame = _frame(connectedHours=[4.0, 245.0, 6.0])
    clean, quarantined = apply_quarantine(frame, spec["quarantine_rules"])
    assert len(clean) == 2
    assert len(quarantined) == 1
    assert "more than one week" in quarantined["quarantine_reason"].iat[0]


def test_non_positive_durations_are_quarantined(spec):
    frame = _frame(connectedHours=[4.0, 0.0, -1.0])
    clean, quarantined = apply_quarantine(frame, spec["quarantine_rules"])
    assert len(clean) == 1
    assert len(quarantined) == 2


def test_a_row_can_collect_multiple_reasons(spec):
    frame = _frame(connectedHours=[4.0, 5.0, -1.0], kWhDelivered=[10.0, 12.0, -5.0])
    _, quarantined = apply_quarantine(frame, spec["quarantine_rules"])
    assert quarantined["quarantine_reason"].iat[0].count(";") == 1


def test_clean_rows_pass_through_untouched(spec):
    frame = _frame()
    clean, quarantined = apply_quarantine(frame, spec["quarantine_rules"])
    assert len(clean) == 3
    assert quarantined.empty


def test_unknown_operator_is_rejected():
    with pytest.raises(ConfigError, match="Unknown quarantine op"):
        apply_quarantine(_frame(), [{"column": "kWhDelivered", "op": "wat", "reason": "x"}])


def test_missing_required_column_blocks(spec, tmp_path):
    from evcharge.config import Secret, Settings

    settings = Settings(
        acn_api_token=Secret("x"), data_dir=tmp_path, artifacts_dir=tmp_path
    )
    frame = _frame().drop(columns=["kWhDelivered"])
    with pytest.raises(ContractBreach, match="Required columns missing"):
        validate(frame, spec, settings, "unit")


def test_mixed_type_site_id_is_a_blocking_failure(spec, tmp_path):
    """Integer siteID must not reach the feature store silently."""
    from evcharge.config import Secret, Settings

    settings = Settings(
        acn_api_token=Secret("x"), data_dir=tmp_path, artifacts_dir=tmp_path
    )
    frame = _frame(siteID=["0001", 2, "0002"])
    report = validate(frame, spec, settings, "unit")
    assert report["passed"] is False
    failed = {c["expectation"] for c in report["blocking_failures"]}
    assert "expect_column_values_to_be_in_set" in failed


def test_station_ids_outside_the_known_namespace_are_blocked(spec, tmp_path):
    from evcharge.config import Secret, Settings

    settings = Settings(
        acn_api_token=Secret("x"), data_dir=tmp_path, artifacts_dir=tmp_path
    )
    report = validate(_frame(stationID=["2-39-123-23", "0001-S001", "0002-S000"]),
                      spec, settings, "unit")
    assert report["passed"] is False
    assert any(
                          check["expectation"] == "expect_station_id_in_site_domain"
        and not check["success"]
        for check in report["blocking_failures"]
    )


def test_contract_reports_timestamp_and_energy_pair_checks(spec):
    from evcharge.validation.contract import run_expectations

    outcome = run_expectations(_frame(), spec)
    pair_checks = [
        check for check in outcome["checks"]
        if check["expectation"] == "expect_column_pair_values_a_to_be_greater_than_b"
    ]
    assert len(pair_checks) == 2
    assert all(check["success"] for check in pair_checks)
    assert any(
        check["expectation"] == "expect_declared_energy_ratio_to_be_plausible"
        and check["success"]
        for check in outcome["checks"]
    )


def test_gross_declared_energy_mismatches_fail_the_contract(spec):
    from evcharge.validation.contract import run_expectations

    outcome = run_expectations(
        _frame(kWhRequested=[1.0, 1.0, 100.0], kWhDelivered=[10.0, 12.0, 8.0]),
        spec,
    )
    energy_check = next(
        check
        for check in outcome["checks"]
        if check["expectation"] == "expect_declared_energy_ratio_to_be_plausible"
    )
    assert not energy_check["success"]
    assert energy_check["unexpected_count"] == 2
