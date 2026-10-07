"""Shared feature builder for training and serving.

This module is the single definition of the feature space. Training and the
serving API both call `transform`, which is what prevents training-serving skew.

Two invariants are enforced rather than assumed:

* **No post-hoc columns.** Anything only observable after a session ends is
  rejected, so a model that cannot actually be served never gets built.
* **No batch-dependent features.** Every feature is a pure function of a single
  row plus the fitted spec, so scoring one row gives the same answer as scoring
  it inside a batch. `tests/test_features.py` asserts this.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a post-hoc column reaches the feature matrix."""


@dataclass
class FeatureSpec:
    """Fitted state persisted alongside the model so serving matches training."""

    categories: dict[str, list[str]]
    feature_names: list[str]
    unknown_category_code: int = -1
    version: str = field(default="1")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def assert_no_leakage(columns: list[str], data_config: dict[str, Any]) -> None:
    post_hoc = set(data_config["post_hoc_columns"])
    identifiers = set(data_config["identifier_columns"])
    offending = sorted((set(columns) & post_hoc) | (set(columns) & identifiers))
    if offending:
        raise LeakageError(
            f"Columns not knowable at plug-in time reached the feature matrix: {offending}"
        )


def _temporal_features(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    times = pd.to_datetime(frame[column])
    hour = times.dt.hour.astype("float64")
    radians = 2 * np.pi * hour / 24.0
    dayofweek = times.dt.dayofweek.astype("float64")

    return pd.DataFrame(
        {
            "arrival_hour": hour,
            "arrival_hour_sin": np.sin(radians),
            "arrival_hour_cos": np.cos(radians),
            "arrival_dayofweek": dayofweek,
            "arrival_is_weekend": (dayofweek >= 5).astype("float64"),
            "arrival_month": times.dt.month.astype("float64"),
        },
        index=frame.index,
    )


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to float, yielding all-NaN if it is absent entirely."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _declared_features(frame: pd.DataFrame) -> pd.DataFrame:
    requested = _numeric_column(frame, "kWhRequested")
    minutes = _numeric_column(frame, "minutesAvailable")

    if "requestedDeparture" in frame.columns and "connectionTime" in frame.columns:
        departure = pd.to_datetime(frame["requestedDeparture"], utc=True, errors="coerce")
        connection = pd.to_datetime(frame["connectionTime"], utc=True, errors="coerce")
        declared_duration = (departure - connection).dt.total_seconds() / 60.0
    else:
        declared_duration = pd.Series(np.nan, index=frame.index, dtype="float64")

    # Guard against the zero-valued requests present in the feed, which would
    # otherwise produce infinities.
    safe_minutes = minutes.where(minutes > 0)

    return pd.DataFrame(
        {
            "declared_duration_minutes": declared_duration.astype("float64"),
            "declared_kwh_per_minute": (requested / safe_minutes).astype("float64"),
            "has_user_inputs": requested.notna().astype("float64"),
        },
        index=frame.index,
    )


def fit(
    frame: pd.DataFrame, feature_config: dict[str, Any], data_config: dict[str, Any]
) -> FeatureSpec:
    """Learn the category vocabulary from the training batch only."""
    categories = {
        column: sorted(frame[column].dropna().astype(str).unique().tolist())
        for column in feature_config["categorical"]
    }
    spec = FeatureSpec(
        categories=categories,
        feature_names=[],
        unknown_category_code=feature_config.get("unknown_category_code", -1),
    )
    spec.feature_names = transform(frame, spec, feature_config, data_config).columns.tolist()
    return spec


def transform(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    feature_config: dict[str, Any],
    data_config: dict[str, Any],
) -> pd.DataFrame:
    """Build the feature matrix. Pure per-row: no batch statistics are used."""
    parts = [
        _temporal_features(frame, feature_config["timestamp_column"]),
        _declared_features(frame),
    ]

    numeric = pd.DataFrame(index=frame.index)
    for column in feature_config["numeric"]:
        numeric[column] = pd.to_numeric(frame.get(column), errors="coerce").astype("float64")
    parts.append(numeric)

    encoded = pd.DataFrame(index=frame.index)
    for column, vocabulary in spec.categories.items():
        lookup = {value: code for code, value in enumerate(vocabulary)}
        encoded[f"{column}_code"] = (
            frame[column].astype(str).map(lookup).fillna(spec.unknown_category_code)
        ).astype("float64")
    parts.append(encoded)

    features = pd.concat(parts, axis=1)
    assert_no_leakage(features.columns.tolist(), data_config)

    if spec.feature_names:
        missing = set(spec.feature_names) - set(features.columns)
        if missing:
            raise LeakageError(f"Features missing at transform time: {sorted(missing)}")
        features = features[spec.feature_names]

    return features


def build_xy(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    feature_config: dict[str, Any],
    data_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    features = transform(frame, spec, feature_config, data_config)
    target = pd.to_numeric(frame[feature_config["target"]], errors="coerce")
    return features, target
