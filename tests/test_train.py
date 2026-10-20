import pandas as pd
import pytest

from evcharge.config import ConfigError, Settings
from evcharge.features.build import FeatureSpec
from evcharge.models import train


def test_training_requires_post_shift_rows(monkeypatch, tmp_path):
    frame = pd.DataFrame({"kWhDelivered": [1.0]})
    monkeypatch.setattr(train, "load_sessions", lambda _settings, _source: frame)
    monkeypatch.setattr(
        train,
        "temporal_split",
        lambda _frame, _split: {
            "train": pd.DataFrame({"kWhDelivered": [1.0]}),
            "validation": pd.DataFrame({"kWhDelivered": [1.0]}),
            "post_shift": pd.DataFrame(columns=["kWhDelivered"]),
        },
    )

    with pytest.raises(ConfigError, match="no labeled post-shift rows"):
        train.run(
            Settings(data_dir=tmp_path, artifacts_dir=tmp_path),
            "sample",
        )


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("sample", "energy", "are not results for this synthetic model"),
        ("raw", "energy", "10% post-shift recovery target remains unmet"),
        ("raw", "dwell", "No drift or retraining recovery result is claimed"),
    ],
)
def test_model_card_scopes_drift_evidence(
    source, target, expected, tmp_path, monkeypatch
):
    monkeypatch.setattr(train, "git_commit", lambda: "0123456789abcdef")
    monkeypatch.setattr(
        Settings, "models_dir", property(lambda _settings: tmp_path / "models")
    )
    settings = Settings()
    metrics = {
        "validation": {"overall": {"mae": 1.0}},
        "post_shift": {"overall": {"mae": 2.0}},
    }

    model_dir = train.persist_best_model(
        settings,
        source,
        target,
        "test-model",
        {"estimator": "stub"},
        FeatureSpec(categories={}, feature_names=["arrival_hour"]),
        metrics,
    )

    card = (model_dir / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert expected in card
    assert "operations_governance.md" in card
