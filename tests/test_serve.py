from fastapi.testclient import TestClient

from evcharge.config import PROJECT_ROOT, load_config
from evcharge.features.build import FeatureSpec
from evcharge.serve.app import create_app
from evcharge.serve.app import request_to_features


def test_health_and_prediction_use_saved_sample_model():
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "source": "sample"}

        prediction = client.post(
            "/predict",
            json={
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
            },
        )

    assert prediction.status_code == 200
    assert prediction.json()["predicted_kWhDelivered"] >= 0


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
