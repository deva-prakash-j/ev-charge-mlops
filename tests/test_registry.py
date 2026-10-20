import json
from types import SimpleNamespace

import pytest

from evcharge.config import ConfigError, Settings
from evcharge.models.registry import (
    evaluate_promotion_gate,
    register_candidate,
    resolve_tracking_uri,
)


def test_tracking_uri_can_be_overridden_for_compose(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    uri = resolve_tracking_uri(
        Settings(artifacts_dir=tmp_path / "artifacts"),
        {"mlflow": {"tracking_uri": "mlruns"}},
    )

    assert uri == "http://mlflow:5000"


def test_tracking_uri_defaults_to_repository_local_store(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = Settings(artifacts_dir=tmp_path / "artifacts")

    uri = resolve_tracking_uri(
        settings, {"mlflow": {"tracking_uri": "mlruns"}}
    )

    assert uri == (tmp_path / "mlruns").resolve().as_uri()


def test_first_candidate_can_bootstrap_only_when_it_beats_both_baselines():
    gate = evaluate_promotion_gate(
        candidate_mae=2.0,
        mean_baseline_mae=5.0,
        ridge_baseline_mae=3.0,
        production_mae=None,
    )

    assert gate["passed"] is True
    assert gate["checks"]["production_regression_within_limit"]["bootstrap"] is True


def test_candidate_must_beat_both_baselines_and_protect_production():
    gate = evaluate_promotion_gate(
        candidate_mae=4.0,
        mean_baseline_mae=5.0,
        ridge_baseline_mae=3.0,
        production_mae=3.0,
        max_production_regression=0.10,
    )

    assert gate["passed"] is False
    assert gate["failed_checks"] == [
        "beats_ridge_baseline",
        "production_regression_within_limit",
    ]


@pytest.mark.parametrize(
    "candidate,mean,ridge,production",
    [
        (float("nan"), 5.0, 3.0, None),
        (2.0, float("inf"), 3.0, None),
        (2.0, 5.0, 3.0, -1.0),
    ],
)
def test_gate_rejects_invalid_mae_values(candidate, mean, ridge, production):
    with pytest.raises(ConfigError, match="finite, non-negative"):
        evaluate_promotion_gate(candidate, mean, ridge, production)


def test_blocked_candidate_is_staged_without_changing_production(tmp_path):
    class FakeClient:
        def __init__(self):
            self.aliases = []
            self.tags = []

        def create_registered_model(self, *args, **kwargs):
            return None

        def create_model_version(self, **kwargs):
            return SimpleNamespace(version=7)

        def set_registered_model_alias(self, name, alias, version):
            self.aliases.append((name, alias, version))

        def set_model_version_tag(self, *args):
            self.tags.append(args)

    client = FakeClient()
    decision_log = tmp_path / "promotion_decisions.jsonl"
    gate = evaluate_promotion_gate(4.0, 5.0, 3.0, None)
    decision = register_candidate(
        client,
        model_name="evcharge-sample-energy",
        source="sample",
        target="energy",
        run_id="run-id",
        feature_spec_artifact="feature_spec_sample.json",
        candidate_mae=4.0,
        mean_baseline_mae=5.0,
        ridge_baseline_mae=3.0,
        production=None,
        gate=gate,
        decision_log=decision_log,
        evidence={
            "git_commit": "commit",
            "data_sha256": "hash",
            "validation_rows": 12,
        },
    )

    assert decision["status"] == "blocked"
    assert client.aliases == [("evcharge-sample-energy", "staging", "7")]
    assert json.loads(decision_log.read_text(encoding="utf-8"))["status"] == "blocked"
