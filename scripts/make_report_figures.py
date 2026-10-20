"""Render aggregate-only report figures from benchmark and drift evidence.

Inputs are metric summaries, never session rows:
- benchmarks/licensed_baseline.json and benchmarks/dwell_baseline.json (tracked)
- artifacts/training_raw.json (python -m evcharge.pipeline --source raw)
- artifacts/drift/detector_evaluation.json (scripts/evaluate_drift.py)
- artifacts/drift/retraining_recovery.json (scripts/evaluate_retraining_recovery.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evcharge.config import PROJECT_ROOT  # noqa: E402

MODEL_LABELS = {
    "baseline_mean": "Mean baseline",
    "baseline_ridge": "Ridge",
    "hist_gradient_boosting": "HistGradientBoosting",
}
SPLIT_LABELS = {
    "train": "Train\n(2018-04..2019-12)",
    "validation": "Validation\n(2020-01..02)",
    "post_shift": "Post-shift\n(2020-04..06)",
}
SITE_LABELS = {"0001": "Site 0001", "0002": "Site 0002", "0019": "Site 0019"}
COLORS = ["#9aa5b1", "#4c78a8", "#f58518"]
ALERT = "#d62728"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing input {path}; regenerate it before plotting.")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, output: Path, name: str) -> Path:
    path = output / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _grouped_bars(ax, metrics: dict, unit: str, title: str) -> None:
    splits = list(SPLIT_LABELS)
    width = 0.26
    for offset, (model, label) in enumerate(MODEL_LABELS.items()):
        values = [metrics[model][split]["mae"] for split in splits]
        xs = [index + (offset - 1) * width for index in range(len(splits))]
        bars = ax.bar(xs, values, width, label=label, color=COLORS[offset])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(len(splits)), [SPLIT_LABELS[s] for s in splits], fontsize=8)
    peak = max(metrics[m][s]["mae"] for m in MODEL_LABELS for s in splits)
    ax.set_ylim(0, peak * 1.3)
    ax.set_ylabel(f"MAE ({unit})")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def model_comparison(output: Path, energy: dict, dwell: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _grouped_bars(axes[0], energy["metrics"], "kWh", "Energy (kWhDelivered)")
    _grouped_bars(axes[1], dwell["metrics"], "hours", "Dwell (connectedHours)")
    axes[0].legend(fontsize=8, loc="upper left", ncol=3)
    fig.suptitle("Chronological MAE: models vs baselines (lower is better)")
    return _save(fig, output, "fig_model_mae.png")


def site_slices(output: Path, training: dict) -> Path:
    results = training["results"]["hist_gradient_boosting"]
    sites = list(SITE_LABELS)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.38
    for offset, split in enumerate(("validation", "post_shift")):
        slices = results[split]["by_siteID"]
        values = [slices[site]["mae"] for site in sites]
        xs = [index + (offset - 0.5) * width for index in range(len(sites))]
        bars = ax.bar(
            xs,
            values,
            width,
            label=SPLIT_LABELS[split].replace("\n", " "),
            color=COLORS[offset + 1],
        )
        labels = [
            f"{value:.1f}\nn={slices[site]['n']}" for value, site in zip(values, sites)
        ]
        ax.bar_label(bars, labels=labels, fontsize=7, padding=2)
    ax.set_xticks(range(len(sites)), [SITE_LABELS[site] for site in sites])
    ax.set_ylabel("Energy MAE (kWh)")
    ax.set_ylim(0, max(results["post_shift"]["by_siteID"][s]["mae"] for s in sites) * 1.3)
    ax.set_title("Energy error by site: site 0019 is under-served")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, output, "fig_site_mae.png")


def drift_timeline(output: Path, detector: dict) -> Path:
    stable = detector["stable_leave_one_month_out"]["months"]
    real = detector["real_batches"]
    months = list(stable) + list(real)
    psi = [stable[m]["target"]["psi"] for m in stable]
    psi += [real[m]["target"]["psi"] for m in real]
    threshold = detector["psi_alert_threshold"]
    colors = [ALERT if value >= threshold else "#4c78a8" for value in psi]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(months, psi, color=colors)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.axhline(threshold, color=ALERT, linestyle="--", linewidth=1)
    ax.text(-0.4, threshold + 0.006, f"alert threshold PSI = {threshold}",
            fontsize=8, color=ALERT)
    ax.axvline(len(stable) - 0.5, color="black", linewidth=0.8, alpha=0.5)
    label_y = threshold * 0.55
    ax.text(len(stable) - 0.4, label_y, "monitored\nbatches", fontsize=8)
    ax.text(len(stable) / 2 - 0.5, label_y, "leave-one-month-out (stable reference)",
            fontsize=8, ha="center")
    ax.set_ylim(0, max(psi) * 1.15)
    ax.set_ylabel("PSI of kWhDelivered vs frozen reference")
    ax.set_title("Drift detector: 0/8 false alarms, April 2020 alert")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, output, "fig_drift_psi.png")


def sensitivity(output: Path, detector: dict) -> Path:
    rows = detector["controlled_energy_shift_sensitivity"]
    shifts = [(row["energy_multiplier"] - 1) * 100 for row in rows]
    psi = [row["psi"] for row in rows]
    threshold = detector["psi_alert_threshold"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(shifts, psi, marker="o", color="#4c78a8")
    for x, y in zip(shifts, psi):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    ax.axhline(threshold, color=ALERT, linestyle="--", linewidth=1, label="alert PSI")
    ax.set_xlabel("Injected multiplicative shift in delivered energy (%)")
    ax.set_ylabel("PSI")
    ax.set_title("Sensitivity: first alert at a +30% energy shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, output, "fig_sensitivity.png")


def recovery(output: Path, energy: dict, recovery_data: dict) -> Path:
    experiments = recovery_data["experiments"]
    baseline = recovery_data["pre_shift_baseline_validation_mae"]
    ceiling = baseline * (1 + recovery_data["recovery_tolerance_fraction"])
    labels = [
        "Frozen model\npost-shift\n(2020-04..06)",
        "Retrained to 2020-06\nscored 2020-09..12",
        "Retrained to 2021-02\nscored 2021-04..05",
    ]
    values = [
        energy["metrics"]["hist_gradient_boosting"]["post_shift"]["mae"],
        experiments["trained_through_2020_06"]["post_shift_mae"],
        experiments["trained_through_2021_02"]["post_shift_mae"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(labels, values, color=["#9aa5b1", "#4c78a8", "#4c78a8"])
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(baseline, color="#2ca02c", linewidth=1, label=f"pre-shift MAE {baseline:.2f}")
    ax.axhline(ceiling, color=ALERT, linestyle="--", linewidth=1,
               label=f"recovery target (+10%) {ceiling:.2f}")
    ax.set_ylabel("Energy MAE (kWh)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Retraining recovery target not met (null result)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, output, "fig_recovery.png")


def importance(output: Path, training: dict) -> Path:
    scores = training["results"]["hist_gradient_boosting"]["permutation_importance"]
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:8]
    names = [name for name, _ in reversed(top)]
    values = [value for _, value in reversed(top)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.barh(names, values, color="#4c78a8")
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel("Validation MAE increase when permuted (kWh)")
    ax.set_title("Driver-declared intent dominates energy prediction")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, output, "fig_importance.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "assets")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    energy = _load(PROJECT_ROOT / "benchmarks" / "licensed_baseline.json")
    dwell = _load(PROJECT_ROOT / "benchmarks" / "dwell_baseline.json")
    training = _load(PROJECT_ROOT / "artifacts" / "training_raw.json")
    detector = _load(PROJECT_ROOT / "artifacts" / "drift" / "detector_evaluation.json")
    recovery_data = _load(
        PROJECT_ROOT / "artifacts" / "drift" / "retraining_recovery.json"
    )
    written = [
        model_comparison(args.output, energy, dwell),
        site_slices(args.output, training),
        drift_timeline(args.output, detector),
        sensitivity(args.output, detector),
        recovery(args.output, energy, recovery_data),
        importance(args.output, training),
    ]
    for path in written:
        print(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


if __name__ == "__main__":
    main()
