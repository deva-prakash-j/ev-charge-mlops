import json
from types import SimpleNamespace

import joblib
import mlflow.sklearn
import pytest
from fastapi.testclient import TestClient

from evcharge.config import PROJECT_ROOT, load_config
from evcharge.features.build import FeatureSpec
import evcharge.serve.app as serve_app
from evcharge.serve.app import create_app
from evcharge.serve.app import request_to_features


PREDICTION_REQUEST = {
    "siteID": "0001",
    "clusterID": "0001",
    "stationID": "unknown-station",
    "connectionTimeLocal": "2020-01-15T08:00:00",
    "connectionTime": "2020-01-15T16:00:00Z",
    "requestedDeparture": "2020-01-15T23:00:00Z",
    "kWhRequested": 20.0,
    "milesRequested": 60.0,
    "minutesAvailable": 420.0,
    "WhPerMile": 300.0,
}


def test_health_and_prediction_use_saved_sample_model():
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "source": "sample"}

        prediction = client.post("/predict", json=PREDICTION_REQUEST)

    assert prediction.status_code == 200
    assert prediction.json()["predicted_kWhDelivered"] >= 0
    assert prediction.json()["model_name"] == "local-sample-energy"
    assert prediction.json()["model_version"] == "local"


def test_prediction_rejects_post_hoc_fields():
    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={
                "siteID": "0001",
                "connectionTimeLocal": "2020-01-15T08:00:00",
                "kWhDelivered": 10,
            },
        )
    assert response.status_code == 422


def test_request_identifier_normalization_matches_ingest():
    spec = FeatureSpec.load(PROJECT_ROOT / "models" / "sample" / "feature_spec.json")
    features = request_to_features(
        {
            "siteID": "2",
            "connectionTimeLocal": "2020-01-15T08:00:00",
            "clusterID": None,
            "stationID": None,
        },
        spec,
        load_config("features"),
        load_config("data"),
    )
    assert features["siteID_code"].iat[0] == spec.categories["siteID"].index("0002")
    assert features["clusterID_code"].iat[0] == spec.unknown_category_code
    assert features["stationID_code"].iat[0] == spec.unknown_category_code


def test_metrics_and_prediction_log_record_features_without_raw_identifiers(tmp_path):
    prediction_log = tmp_path / "served" / "predictions.jsonl"
    with TestClient(create_app(prediction_log_path=prediction_log)) as client:
        response = client.post("/predict", json=PREDICTION_REQUEST)
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert metrics.status_code == 200
    assert "evcharge_predictions_total" in metrics.text
    assert "evcharge_prediction_kwh_bucket" in metrics.text
    record = json.loads(prediction_log.read_text(encoding="utf-8").strip())
    assert record["predicted_kwh"] == response.json()["predicted_kWhDelivered"]
    assert record["model_name"] == "local-sample-energy"
    assert "features" in record
    assert "siteID" not in record
    assert "stationID" not in record


def test_registry_mode_loads_production_alias(monkeypatch):
    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "http://mlflow.test"

        def get_model_version_by_alias(self, name, alias):
            assert name == "evcharge-sample-energy"
            assert alias == "production"
            return SimpleNamespace(
                version="12",
                run_id="run-id",
                tags={
                    "feature_spec_artifact": "feature_spec_sample.json",
                    "source": "sample",
                },
            )

        def download_artifacts(self, run_id, artifact_path):
            assert run_id == "run-id"
            assert artifact_path == "feature_spec_sample.json"
            return str(PROJECT_ROOT / "models" / "sample" / "feature_spec.json")

    monkeypatch.setattr(
        serve_app, "resolve_tracking_uri", lambda *_args: "http://mlflow.test"
    )
    monkeypatch.setattr(serve_app, "MlflowClient", FakeClient)
    monkeypatch.setattr(serve_app.mlflow, "set_tracking_uri", lambda _uri: None)

    loaded_uris = []

    def load_production_model(uri):
        loaded_uris.append(uri)
        return joblib.load(PROJECT_ROOT / "models" / "sample" / "model.joblib")

    # mlflow.sklearn starts as a lazy proxy that replaces itself on first use, so
    # patching the proxy would leave the app calling the real registry loader.
    monkeypatch.setattr(mlflow.sklearn, "load_model", load_production_model)

    with TestClient(
        create_app(registry_model_name="evcharge-sample-energy")
    ) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=PREDICTION_REQUEST)

    assert health.json() == {
        "status": "ok",
        "source": "sample",
        "model_name": "evcharge-sample-energy",
        "model_version": "12",
    }
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == "12"
    assert loaded_uris == ["models:/evcharge-sample-energy@production"]
