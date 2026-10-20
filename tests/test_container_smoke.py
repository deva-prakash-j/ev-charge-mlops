"""Optional smoke tests for an already-running Docker Compose API."""

import json
import os
from urllib.request import Request, urlopen

import pytest

from evcharge.config import PROJECT_ROOT


def test_container_health_prediction_and_metrics():
    base_url = os.getenv("EVCHARGE_API_URL")
    if not base_url:
        pytest.skip("Set EVCHARGE_API_URL to run the Compose smoke test.")

    with urlopen(f"{base_url.rstrip('/')}/health", timeout=10) as response:
        health = json.load(response)
    assert health["status"] == "ok"
    assert health["model_name"] == "evcharge-sample-energy"

    payload = (
        PROJECT_ROOT / "examples" / "predict_request.json"
    ).read_bytes()
    request = Request(
        f"{base_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        prediction = json.load(response)
    assert prediction["predicted_kWhDelivered"] >= 0
    assert prediction["model_name"] == health["model_name"]
    assert prediction["model_version"] == health["model_version"]

    with urlopen(f"{base_url.rstrip('/')}/metrics", timeout=10) as response:
        metrics = response.read().decode("utf-8")
    assert "evcharge_predictions_total" in metrics
    assert "evcharge_prediction_kwh_bucket" in metrics
