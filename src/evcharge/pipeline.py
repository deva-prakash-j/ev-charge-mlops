"""Single entry point for the end-to-end pipeline.

    python -m evcharge.pipeline --source sample

Stages run in order and the run halts on a blocking contract breach, so a
schema violation can never reach training. Every stage records its outcome to
artifacts/pipeline_<source>.json.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable

from evcharge.config import ConfigError, Settings, get_settings, load_config

logger = logging.getLogger(__name__)


class StageFailure(RuntimeError):
    """Raised when a stage fails in a way that must stop the pipeline."""


def stage_fetch(settings: Settings, source: str) -> dict[str, Any]:
    from evcharge.ingest.fetch_sessions import main as fetch_main

    if source != "raw":
        return {"skipped": "fetch only applies to --source raw"}
    if fetch_main([]) != 0:
        raise StageFailure("Download from ACN-Data failed.")
    return {"downloaded": True}


def stage_normalise(settings: Settings, source: str) -> dict[str, Any]:
    from evcharge.ingest.normalise import run as normalise_run

    frame = normalise_run(settings, load_config("data"), source)
    return {
        "rows": len(frame),
        "sites": sorted(frame["siteID"].unique().tolist()),
        "months": [str(frame["year_month"].min()), str(frame["year_month"].max())],
    }


def stage_contract(settings: Settings, source: str) -> dict[str, Any]:
    from evcharge.validation.contract import load_normalised, validate

    spec = load_config("contract")
    report = validate(load_normalised(settings, source), spec, settings, source)
    if not report["passed"]:
        failures = [
            f"{c['expectation']}({c['column']})" for c in report["blocking_failures"]
        ]
        raise StageFailure(f"Data contract breached: {failures}")
    return {
        "rows_clean": report["rows_clean"],
        "rows_quarantined": report["rows_quarantined"],
        "expectations": report["expectations"]["statistics"],
    }


def stage_train(settings: Settings, source: str) -> dict[str, Any]:
    from evcharge.models.train import run as train_run

    report = train_run(settings, source)
    validation = {
        model: splits["validation"]["overall"]["mae"]
        for model, splits in report["results"].items()
    }
    best = min(validation, key=validation.get)
    return {
        "rows": report["rows"],
        "validation_mae": {k: round(v, 4) for k, v in validation.items()},
        "best_model": best,
        "git_commit": report["git_commit"][:12],
    }


Stage = Callable[[Settings, str], dict[str, Any]]

STAGES: dict[str, Stage] = {
    "fetch": stage_fetch,
    "normalise": stage_normalise,
    "contract": stage_contract,
    "train": stage_train,
}

# fetch is opt-in: it needs credentials and hits the network.
DEFAULT_STAGES = ("normalise", "contract", "train")


def run_pipeline(
    settings: Settings, source: str, stages: tuple[str, ...]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source": source,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": {},
    }

    for name in stages:
        logger.info("--- %s ---", name)
        started = time.perf_counter()
        try:
            result = STAGES[name](settings, source)
        except StageFailure as exc:
            summary["stages"][name] = {"status": "failed", "error": str(exc)}
            summary["passed"] = False
            _persist(settings, source, summary)
            raise
        summary["stages"][name] = {
            "status": "ok",
            "seconds": round(time.perf_counter() - started, 2),
            **result,
        }

    summary["passed"] = True
    _persist(settings, source, summary)
    return summary


def _persist(settings: Settings, source: str, summary: dict[str, Any]) -> None:
    import json

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / f"pipeline_{source}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline.")
    parser.add_argument(
        "--source",
        choices=("sample", "raw"),
        default="sample",
        help="'sample' uses the committed fixture and needs no credentials.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download from the ACN-Data API first. Requires ACN_API_TOKEN.",
    )
    args = parser.parse_args(argv)

    # Set before Great Expectations or MLflow load, so their progress bars and
    # upstream deprecation warnings stay out of the pipeline log.
    os.environ.setdefault("TQDM_DISABLE", "1")
    warnings.filterwarnings("ignore")

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    # mlflow calls logging.captureWarnings(True), which routes upstream
    # deprecation warnings into the log regardless of the warnings filter.
    for noisy in ("mlflow", "great_expectations", "py.warnings"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    stages = (("fetch",) if args.fetch else ()) + DEFAULT_STAGES

    try:
        settings = get_settings()
        summary = run_pipeline(settings, args.source, stages)
    except StageFailure as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 2

    print(f"\n=== pipeline complete ({args.source}) ===")
    for name, result in summary["stages"].items():
        detail = {
            k: v for k, v in result.items() if k not in {"status", "seconds"}
        }
        print(f"  {name:<10} {result['status']:<4} {result.get('seconds', 0):>6.2f}s  {detail}")
    print(f"\n[ok]   report -> {settings.artifacts_dir / f'pipeline_{args.source}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
