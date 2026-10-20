"""Run the deliberately degraded model proof and verify Production is unchanged."""

from __future__ import annotations

import json

import mlflow
from mlflow import MlflowClient

from evcharge.config import get_settings, load_config
from evcharge.models.registry import (
    prove_degraded_candidate,
    registered_model_name,
    resolve_tracking_uri,
)


def main() -> int:
    settings = get_settings()
    model_config = load_config("model")
    tracking_uri = resolve_tracking_uri(settings, model_config)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = registered_model_name(
        "sample", "energy", model_config["registry"]["registered_model_prefix"]
    )

    promoted_versions = {}
    production_before = {}
    for target in ("energy", "dwell"):
        target_name = registered_model_name(
            "sample", target, model_config["registry"]["registered_model_prefix"]
        )
        report_name = (
            "training_sample.json"
            if target == "energy"
            else "training_dwell_sample.json"
        )
        report_path = settings.artifacts_dir / report_name
        report = json.loads(report_path.read_text(encoding="utf-8"))
        promotion = report.get("promotion", {})
        if promotion.get("status") != "promoted" or not promotion.get(
            "gate", {}
        ).get("passed"):
            raise RuntimeError(
                f"The {target} model did not pass the promotion gate: "
                f"{promotion.get('gate', {}).get('failed_checks', [])}"
            )

        production_version = str(
            client.get_model_version_by_alias(target_name, "production").version
        )
        candidate_version = str(promotion["candidate_version"])
        if production_version != candidate_version:
            raise RuntimeError(
                f"The {target} Production alias points to version "
                f"{production_version}, not promoted candidate {candidate_version}."
            )
        promoted_versions[target] = candidate_version
        production_before[target] = production_version

    decision = prove_degraded_candidate(settings, "sample", "energy")
    production_after = {
        target: str(
            client.get_model_version_by_alias(
                registered_model_name(
                    "sample", target, model_config["registry"]["registered_model_prefix"]
                ),
                "production",
            ).version
        )
        for target in ("energy", "dwell")
    }

    if decision["status"] != "blocked" or decision["gate"]["passed"]:
        raise RuntimeError("The deliberately degraded candidate was not blocked.")
    if not decision["gate"]["failed_checks"]:
        raise RuntimeError("The blocked candidate has no failed gate checks.")
    if production_after != production_before:
        raise RuntimeError(
            f"The degraded candidate changed Production from version "
            f"{production_before} to {production_after}."
        )

    evidence = {
        "model_name": model_name,
        "production_before": production_before,
        "production_after": production_after,
        "promoted_versions": promoted_versions,
        "degraded_candidate_version": decision["candidate_version"],
        "degraded_candidate_run_id": decision["candidate_run_id"],
        "candidate_validation_mae": decision["candidate_validation_mae"],
        "failed_checks": decision["gate"]["failed_checks"],
        "status": decision["status"],
    }
    evidence_path = settings.artifacts_dir / "ci_promotion_gate.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"[ok]   degraded candidate {model_name}/{decision['candidate_version']} "
        f"was blocked"
    )
    print(
        f"[ok]   Production aliases stayed at {production_after}; "
        f"failed checks: {', '.join(decision['gate']['failed_checks'])}"
    )
    print(f"[ok]   evidence -> {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
