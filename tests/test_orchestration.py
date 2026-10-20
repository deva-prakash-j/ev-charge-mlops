import json

import pandas as pd
import pytest

from evcharge.config import ConfigError, Settings
from evcharge.orchestration import flows


def test_training_flow_delegates_to_contract_gated_pipeline(monkeypatch):
    from evcharge.pipeline import DEFAULT_STAGES

    settings = object()
    calls = []
    monkeypatch.setattr(flows, "get_settings", lambda: settings)
    monkeypatch.setattr(
        "evcharge.pipeline.run_pipeline",
        lambda actual_settings, source, stages, stage_overrides=None: (
            calls.append((actual_settings, source, stages, stage_overrides))
            or {"passed": True}
        ),
    )

    result = flows.training_flow.fn(source="raw", fetch=True)

    assert result == {"passed": True}
    assert calls == [(settings, "raw", ("fetch",) + DEFAULT_STAGES, None)]


def test_training_flow_routes_retraining_split_to_both_targets(monkeypatch):
    from evcharge.models import train

    settings = object()
    split = {
        "train_end": "2020-02-29",
        "validation_end": "2020-03-31",
        "post_shift_start": "2020-04-01",
        "post_shift_end": "2020-04-30",
    }
    train_calls = []

    def fake_train(_settings, source, target="energy", split_override=None):
        train_calls.append((source, target, split_override))
        return {
            "rows": {"train": 3, "validation": 2, "post_shift": 1},
            "git_commit": "abcdef0123456789",
            "results": {
                "baseline_mean": {
                    "validation": {"overall": {"mae": 3.0}},
                    "post_shift": {"overall": {"mae": 4.0}},
                },
                "hist_gradient_boosting": {
                    "validation": {"overall": {"mae": 2.0}},
                    "post_shift": {"overall": {"mae": 3.0}},
                },
            },
        }

    def run_pipeline(actual_settings, source, _stages, stage_overrides=None):
        assert actual_settings is settings
        return stage_overrides["train"](actual_settings, source)

    monkeypatch.setattr(flows, "get_settings", lambda: settings)
    monkeypatch.setattr("evcharge.pipeline.run_pipeline", run_pipeline)
    monkeypatch.setattr(train, "run", fake_train)

    result = flows.training_flow.fn(source="raw", split_override=split)

    assert result["post_shift_mae"]["hist_gradient_boosting"] == 3.0
    assert train_calls == [
        ("raw", "energy", split),
        ("raw", "dwell", split),
    ]


def test_monthly_flow_runs_fetch_optional_normalise_and_batch_check(
    tmp_path, monkeypatch
):
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path)
    flows._save_monitor_state(
        tmp_path / "monitoring" / "monthly_raw.json",
        {
            "version": flows.STATE_VERSION,
            "source": "raw",
            "batches": {"2020-03": {"fingerprint": "batch-1"}},
        },
    )
    monkeypatch.setattr(flows, "get_settings", lambda: settings)
    calls = []
    monkeypatch.setattr(
        flows, "fetch_raw_data", lambda: calls.append("fetch") or {"downloaded": True}
    )
    monkeypatch.setattr(
        flows,
        "normalise_source",
        lambda source: calls.append(("normalise", source)) or {"rows": 2},
    )
    monkeypatch.setattr(
        flows,
        "validate_monthly_batches",
        lambda source: calls.append(("validate", source)) or [{"month": "2020-03"}],
    )
    monkeypatch.setattr(
        flows,
        "score_and_export_drift",
        lambda source, months: calls.append(("score", source, months))
        or [{"batch_month": "2020-03", "drift_detected": False}],
    )
    monkeypatch.setattr(
        flows,
        "_monthly_partitions",
        lambda _settings, _source: {"2020-03": []},
    )

    result = flows.monthly_monitoring_flow.fn(source="raw", fetch=True)

    assert calls == [
        "fetch",
        ("normalise", "raw"),
        ("validate", "raw"),
        ("score", "raw", ["2020-03"]),
    ]
    assert result["new_or_changed_batches"] == [{"month": "2020-03"}]


def test_synthetic_monitor_scores_drift_without_fetching(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path)
    flows._save_monitor_state(
        tmp_path / "monitoring" / "monthly_synthetic.json",
        {
            "version": flows.STATE_VERSION,
            "source": "synthetic",
            "batches": {"2020-04": {"fingerprint": "batch-1"}},
        },
    )
    monkeypatch.setattr(flows, "get_settings", lambda: settings)
    calls = []
    monkeypatch.setattr(flows, "fetch_raw_data", lambda: calls.append("fetch"))
    monkeypatch.setattr(
        flows, "normalise_source", lambda source: calls.append(("normalise", source)) or {}
    )
    monkeypatch.setattr(flows, "validate_monthly_batches", lambda _source: [])
    monkeypatch.setattr(
        flows,
        "score_and_export_drift",
        lambda source, months: calls.append(("score", source, months))
        or [{"batch_month": "2020-04", "drift_detected": False}],
    )
    monkeypatch.setattr(
        flows, "_monthly_partitions", lambda _settings, _source: {"2020-04": []}
    )

    flows.monthly_monitoring_flow.fn(source="synthetic", fetch=True)

    assert calls == [("normalise", "synthetic"), ("score", "synthetic", ["2020-04"])]


def test_sample_monitor_validates_batches_without_drift_scoring(monkeypatch):
    calls = []
    monkeypatch.setattr(flows, "fetch_raw_data", lambda: calls.append("fetch"))
    monkeypatch.setattr(flows, "normalise_source", lambda _source: {})
    monkeypatch.setattr(flows, "validate_monthly_batches", lambda _source: [])
    monkeypatch.setattr(
        flows,
        "score_and_export_drift",
        lambda source, months: calls.append("score"),
    )

    result = flows.monthly_monitoring_flow.fn(source="sample", fetch=True)

    assert calls == []
    assert result["drift_reports"] == []


def test_monthly_batches_validate_once_and_recheck_changed_data(
    tmp_path, monkeypatch
):
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path)
    partition = (
        settings.interim_dir
        / "raw"
        / "sessions"
        / "site=0001"
        / "month=2020-03"
        / "sessions.parquet"
    )
    partition.parent.mkdir(parents=True)
    pd.DataFrame({"kWhDelivered": [10.0]}).to_parquet(partition, index=False)
    validations = []

    monkeypatch.setattr(flows, "load_config", lambda _name: {"contract": True})
    monkeypatch.setattr(
        "evcharge.validation.contract.validate",
        lambda frame, spec, _settings, source: (
            validations.append((len(frame), source))
            or {"passed": True, "blocking_failures": []}
        ),
    )

    first = flows.check_monthly_batches(settings, "raw")
    second = flows.check_monthly_batches(settings, "raw")
    assert [batch["month"] for batch in first] == ["2020-03"]
    assert second == []
    assert validations == [(1, "raw_2020-03")]

    pd.DataFrame({"kWhDelivered": [10.0, 11.0]}).to_parquet(partition, index=False)
    changed = flows.check_monthly_batches(settings, "raw")
    assert [batch["rows"] for batch in changed] == [2]
    assert len(validations) == 2

    state = json.loads(
        (settings.artifacts_dir / "monitoring" / "monthly_raw.json").read_text()
    )
    assert state["batches"]["2020-03"]["rows"] == 2


def test_monthly_flow_retries_unscored_batches_and_retrains_alert_once(
    tmp_path, monkeypatch
):
    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path)
    state_path = tmp_path / "monitoring" / "monthly_raw.json"
    flows._save_monitor_state(
        state_path,
        {
            "version": flows.STATE_VERSION,
            "source": "raw",
            "batches": {"2020-04": {"fingerprint": "batch-1"}},
        },
    )
    monkeypatch.setattr(flows, "get_settings", lambda: settings)
    monkeypatch.setattr(flows, "normalise_source", lambda _source: {"rows": 1})
    monkeypatch.setattr(flows, "validate_monthly_batches", lambda _source: [])
    monkeypatch.setattr(
        flows, "_monthly_partitions", lambda _settings, _source: {"2020-04": []}
    )
    alert_report = {
        "batch_month": "2020-04",
        "drift_detected": True,
        "scores": [{"column": "kWhDelivered", "psi_alert": True}],
    }
    score_calls = []
    retrain_calls = []

    def score_and_export(source, months):
        score_calls.append((source, months))
        report_path = tmp_path / "drift" / "raw_2020-04.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(alert_report), encoding="utf-8")
        return [alert_report]

    monkeypatch.setattr(flows, "score_and_export_drift", score_and_export)
    monkeypatch.setattr(
        flows,
        "retrain_on_drift_alert",
        lambda alert, source: retrain_calls.append((alert, source))
        or {"triggered": True},
    )

    first = flows.monthly_monitoring_flow.fn(source="raw", fetch=False)
    second = flows.monthly_monitoring_flow.fn(source="raw", fetch=False)

    assert first["retraining"] == {"triggered": True}
    assert second["retraining"] is None
    assert score_calls == [("raw", ["2020-04"])]
    assert len(retrain_calls) == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["batches"]["2020-04"]["scored_fingerprint"] == "batch-1"
    assert state["batches"]["2020-04"]["retrained_fingerprint"] == "batch-1"


def test_monthly_monitor_state_migrates_version_one(tmp_path):
    state_path = tmp_path / "monthly_raw.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": "raw",
                "batches": {"2020-03": {"fingerprint": "batch-1"}},
            }
        ),
        encoding="utf-8",
    )

    state = flows._load_monitor_state(state_path, "raw")

    assert state["version"] == flows.STATE_VERSION
    assert state["batches"]["2020-03"]["fingerprint"] == "batch-1"


def test_contract_failure_does_not_advance_monthly_state(tmp_path, monkeypatch):
    from evcharge.validation.contract import ContractBreach

    settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path)
    partition = (
        settings.interim_dir
        / "raw"
        / "sessions"
        / "site=0001"
        / "month=2020-03"
        / "sessions.parquet"
    )
    partition.parent.mkdir(parents=True)
    pd.DataFrame({"kWhDelivered": [10.0]}).to_parquet(partition, index=False)

    monkeypatch.setattr(flows, "load_config", lambda _name: {})
    monkeypatch.setattr(
        "evcharge.validation.contract.validate",
        lambda *_args: {"passed": False, "blocking_failures": ["breach"]},
    )
    with pytest.raises(ContractBreach, match="2020-03"):
        flows.check_monthly_batches(settings, "raw")
    assert not (settings.artifacts_dir / "monitoring" / "monthly_raw.json").exists()


def test_drift_hook_triggers_training_only_on_boolean_alert(monkeypatch):
    calls = []
    monkeypatch.setattr(
        flows,
        "training_flow",
        lambda **kwargs: calls.append(kwargs) or {"passed": True},
    )

    skipped = flows.retrain_on_drift_alert.fn({"drift_detected": False})
    triggered = flows.retrain_on_drift_alert.fn(
        {"drift_detected": True, "batch_month": "2020-03"}
    )
    split = flows.retraining_split_for_month("2020-03")
    assert skipped == {"triggered": False, "reason": "no drift detected"}
    assert triggered == {
        "triggered": True,
        "split": split,
        "training": {"passed": True},
    }
    assert calls == [{"source": "raw", "fetch": False, "split_override": split}]

    with pytest.raises(ValueError, match="boolean"):
        flows.retrain_on_drift_alert.fn({"drift_detected": 1})
    with pytest.raises(ValueError, match="batch_month"):
        flows.retrain_on_drift_alert.fn({"drift_detected": True})
    assert split == {
        "train_end": "2020-01-31",
        "validation_end": "2020-02-29",
        "post_shift_start": "2020-03-01",
        "post_shift_end": "2020-03-31",
    }


def test_monthly_deployment_registers_utc_schedule(monkeypatch):
    from prefect.client.schemas.schedules import CronSchedule

    captured = {}
    monkeypatch.setenv("PREFECT_API_URL", "http://127.0.0.1:4200/api")
    monkeypatch.setattr(
        flows.monthly_monitoring_flow, "serve", lambda **kwargs: captured.update(kwargs)
    )
    flows.serve_monthly_monitor()

    assert captured["name"] == "monthly-raw-batch-monitor"
    assert captured["schedule"] == CronSchedule(
        cron="0 2 1 * *", timezone="UTC"
    )
    assert captured["parameters"] == {"source": "raw", "fetch": True}


def test_monthly_deployment_requires_prefect_api_setting(monkeypatch):
    monkeypatch.setattr(flows.PREFECT_API_URL, "value", lambda: None)
    with pytest.raises(ConfigError, match="PREFECT_API_URL"):
        flows.serve_monthly_monitor()
