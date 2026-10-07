"""Regression metrics and slice-wise evaluation.

Aggregate metrics hide the degradation this project exists to detect, so every
evaluation is also reported per site and per regime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(truth) & np.isfinite(predicted)
    truth, predicted = truth[mask], predicted[mask]

    if truth.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "median_ae": float("nan"), "r2": float("nan"), "smape": float("nan")}

    error = predicted - truth
    # sMAPE rather than MAPE: sessions delivering ~0.5 kWh make MAPE explode.
    denominator = (np.abs(truth) + np.abs(predicted)) / 2.0
    smape = np.mean(np.abs(error) / np.where(denominator > 0, denominator, np.nan)) * 100

    total_variance = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum(error**2) / total_variance if total_variance > 0 else float("nan")

    return {
        "n": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_ae": float(np.median(np.abs(error))),
        "r2": float(r2),
        "smape": float(smape),
    }


def evaluate_slices(
    frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    breakdowns: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": regression_metrics(y_true, y_pred)}

    for column in breakdowns:
        if column not in frame.columns:
            continue
        per_value: dict[str, dict[str, float]] = {}
        for value, index in frame.groupby(column).groups.items():
            positions = frame.index.get_indexer(index)
            per_value[str(value)] = regression_metrics(
                y_true.iloc[positions], y_pred[positions]
            )
        result[f"by_{column}"] = per_value

    return result


def format_metrics(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label:<22} n={metrics['n']:>6,}  MAE={metrics['mae']:>6.3f}  "
        f"RMSE={metrics['rmse']:>6.3f}  R2={metrics['r2']:>6.3f}  "
        f"sMAPE={metrics['smape']:>5.1f}%"
    )
