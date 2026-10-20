import pandas as pd
import pytest

from evcharge.config import ConfigError, Settings
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
