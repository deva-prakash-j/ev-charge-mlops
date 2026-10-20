from fastapi.testclient import TestClient

from evcharge.serve.app import create_app


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
