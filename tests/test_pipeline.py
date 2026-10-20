"""The pipeline must halt on a contract breach rather than training on bad data."""

import pytest

from evcharge.config import Secret, Settings
from evcharge.pipeline import (
    DEFAULT_STAGES,
    STAGES,
    StageFailure,
    run_pipeline,
    stage_train,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(acn_api_token=Secret("x"), data_dir=tmp_path, artifacts_dir=tmp_path)


def test_default_stage_order_is_validate_before_train():
    assert DEFAULT_STAGES == ("normalise", "contract", "train")
    assert DEFAULT_STAGES.index("contract") < DEFAULT_STAGES.index("train")


def test_fetch_is_not_run_by_default():
    """Fetch needs credentials and hits the network, so it must be opt-in."""
    assert "fetch" not in DEFAULT_STAGES
    assert "fetch" in STAGES


def test_contract_breach_stops_before_training(settings, monkeypatch):
    trained = []

    def failing_contract(_settings, _source):
        raise StageFailure("Data contract breached: ['expect_column_values_to_be_in_set']")

    monkeypatch.setitem(STAGES, "normalise", lambda s, src: {"rows": 10})
    monkeypatch.setitem(STAGES, "contract", failing_contract)
    monkeypatch.setitem(STAGES, "train", lambda s, src: trained.append(src) or {})

    with pytest.raises(StageFailure, match="contract breached"):
        run_pipeline(settings, "unit", DEFAULT_STAGES)

    assert trained == [], "training ran despite a blocking contract failure"


def test_failed_run_is_recorded(settings, monkeypatch):
    import json

    monkeypatch.setitem(STAGES, "normalise", lambda s, src: {"rows": 1})
    monkeypatch.setitem(
        STAGES, "contract", lambda s, src: (_ for _ in ()).throw(StageFailure("boom"))
    )

    with pytest.raises(StageFailure):
        run_pipeline(settings, "unit", DEFAULT_STAGES)

    summary = json.loads((settings.artifacts_dir / "pipeline_unit.json").read_text())
    assert summary["passed"] is False
    assert summary["stages"]["contract"]["status"] == "failed"
    assert "train" not in summary["stages"]


def test_unexpected_stage_error_is_recorded(settings, monkeypatch):
    import json

    monkeypatch.setitem(STAGES, "normalise", lambda s, src: {"rows": 1})
    monkeypatch.setitem(
        STAGES,
        "contract",
        lambda s, src: (_ for _ in ()).throw(ValueError("unexpected defect")),
    )

    with pytest.raises(ValueError, match="unexpected defect"):
        run_pipeline(settings, "unit", DEFAULT_STAGES)

    summary = json.loads((settings.artifacts_dir / "pipeline_unit.json").read_text())
    assert summary["passed"] is False
    assert summary["stages"]["contract"]["status"] == "failed"
    assert summary["stages"]["contract"]["error_type"] == "ValueError"
    assert summary["stages"]["contract"]["error"] == "unexpected defect"


def test_successful_run_records_every_stage(settings, monkeypatch):
    for name in DEFAULT_STAGES:
        monkeypatch.setitem(STAGES, name, lambda s, src, n=name: {"did": n})

    summary = run_pipeline(settings, "unit", DEFAULT_STAGES)

    assert summary["passed"] is True
    assert list(summary["stages"]) == list(DEFAULT_STAGES)
    assert all(v["status"] == "ok" for v in summary["stages"].values())


def test_train_stage_builds_energy_and_dwell_models(settings, monkeypatch):
    from evcharge.models import train

    calls = []

    def fake_run(_settings, source, target="energy"):
        calls.append((source, target))
        return {
            "rows": {"train": 3, "validation": 2, "post_shift": 1},
            "git_commit": "abcdef0123456789",
            "results": {
                "baseline_mean": {"validation": {"overall": {"mae": 3.0}}},
                "hist_gradient_boosting": {
                    "validation": {"overall": {"mae": 2.0}}
                },
            },
        }

    monkeypatch.setattr(train, "run", fake_run)
    report = stage_train(settings, "raw")

    assert calls == [("raw", "energy"), ("raw", "dwell")]
    assert report["best_model"] == "hist_gradient_boosting"
    assert report["best_dwell_model"] == "hist_gradient_boosting"
    assert report["dwell_validation_mae"]["baseline_mean"] == 3.0
