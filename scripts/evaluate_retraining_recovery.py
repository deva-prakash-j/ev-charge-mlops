"""Measure chronological retraining recovery without registering candidates."""

from __future__ import annotations

import json

import numpy as np

from evcharge.config import PROJECT_ROOT, get_settings, load_config
from evcharge.features.build import build_xy, fit as fit_features
from evcharge.models.evaluate import regression_metrics
from evcharge.models.train import build_models, load_sessions, temporal_split
from evcharge.validation.contract import apply_quarantine

PRE_SHIFT_BASELINE_MAE = "hist_gradient_boosting"
RECOVERY_TOLERANCE = 0.10
WINDOWS = {
    "trained_through_2020_06": {
        "train_end": "2020-06-30",
        "validation_end": "2020-08-31",
        "post_shift_start": "2020-09-01",
        "post_shift_end": "2020-12-31",
    },
    "trained_through_2021_02": {
        "train_end": "2021-02-28",
        "validation_end": "2021-03-31",
        "post_shift_start": "2021-04-01",
        "post_shift_end": "2021-05-31",
    },
}


def _evaluate_window(frame, split_config, configs, target):
    splits = temporal_split(frame, split_config)
    if any(splits[name].empty for name in ("train", "validation", "post_shift")):
        raise ValueError(f"Recovery split has an empty window: {split_config}")

    spec = fit_features(splits["train"], configs["features"], configs["data"])
    prepared = {
        name: build_xy(
            part,
            spec,
            configs["features"],
            configs["data"],
            target=target,
        )
        for name, part in splits.items()
    }
    model_scores = {}
    fitted = {}
    for name, estimator in build_models(configs["model"]).items():
        x_train, y_train = prepared["train"]
        estimator.fit(x_train, y_train)
        fitted[name] = estimator
        validation_x, validation_y = prepared["validation"]
        validation = regression_metrics(
            validation_y, np.asarray(estimator.predict(validation_x))
        )
        model_scores[name] = {
            "validation_rows": validation["n"],
            "validation_mae": validation["mae"],
        }

    selected = min(model_scores, key=lambda name: model_scores[name]["validation_mae"])
    post_x, post_y = prepared["post_shift"]
    post = regression_metrics(
        post_y, np.asarray(fitted[selected].predict(post_x))
    )
    return {
        "split": split_config,
        "rows": {name: len(part) for name, part in splits.items()},
        "models": model_scores,
        "selected_model": selected,
        "post_shift_rows": post["n"],
        "post_shift_mae": post["mae"],
    }


def main() -> int:
    settings = get_settings()
    configs = {
        name: load_config(name) for name in ("data", "features", "model", "contract")
    }
    target = configs["features"]["target"]
    frame = load_sessions(settings, "raw")
    frame, _ = apply_quarantine(frame, configs["contract"]["quarantine_rules"])
    frame = frame[frame[target].notna()].reset_index(drop=True)

    baseline_path = PROJECT_ROOT / "benchmarks" / "licensed_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_mae = baseline["metrics"][PRE_SHIFT_BASELINE_MAE]["validation"]["mae"]
    max_recovered_mae = baseline_mae * (1 + RECOVERY_TOLERANCE)
    experiments = {
        name: _evaluate_window(frame, split, configs, target)
        for name, split in WINDOWS.items()
    }
    for experiment in experiments.values():
        experiment["recovery_ceiling_mae"] = max_recovered_mae
        experiment["within_10_percent_of_pre_shift_baseline"] = (
            experiment["post_shift_mae"] <= max_recovered_mae
        )

    report = {
        "source": "raw",
        "target": target,
        "pre_shift_baseline_validation_mae": baseline_mae,
        "recovery_tolerance_fraction": RECOVERY_TOLERANCE,
        "experiments": experiments,
    }
    output = settings.artifacts_dir / "drift" / "retraining_recovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Recovery report saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
