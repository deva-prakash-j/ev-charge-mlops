import numpy as np
import pandas as pd

from evcharge.features.build import FeatureSpec
from evcharge.monitoring.drift import (
    PSI_ALERT_THRESHOLD,
    _prediction_frame,
    _score_metrics,
    push_drift_metrics,
    score_new_months,
)


def test_evidently_reports_psi_ks_and_jensen_shannon_for_numeric_features():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"feature": rng.normal(10, 1, 1000)})
    current = pd.DataFrame({"feature": rng.normal(14, 1, 1000)})

    scores, result = _score_metrics(reference, current)

    feature = scores[0]
    assert feature["psi"] > PSI_ALERT_THRESHOLD
    assert feature["ks_statistic"] > 0.5
    assert feature["ks_pvalue"] < 0.01
    assert feature["js_distance"] > 0
    assert result["summary"]["drift_detected"] is True


def test_categorical_features_use_psi_and_js_without_invalid_ks():
    reference = pd.DataFrame({"siteID_code": ["0001", "0002"] * 100})
    current = pd.DataFrame({"siteID_code": ["0001"] * 150 + ["0019"] * 50})

    scores, _ = _score_metrics(reference, current)

    assert scores[0]["psi"] > PSI_ALERT_THRESHOLD
    assert scores[0]["js_distance"] > 0
    assert scores[0]["ks_statistic"] is None
    assert scores[0]["ks_pvalue"] is None


def test_insufficient_feature_scores_still_report_missing_coverage():
    reference = pd.DataFrame(
        {
            "feature": np.arange(10, dtype="float64"),
            "sparse": [1.0] + [np.nan] * 9,
        }
    )
    current = pd.DataFrame(
        {
            "feature": np.arange(10, dtype="float64"),
            "sparse": [1.0, 2.0] + [np.nan] * 8,
        }
    )

    scores, _ = _score_metrics(reference, current)
    sparse = next(score for score in scores if score["column"] == "sparse")

    assert sparse["status"] == "insufficient_non_missing_values"
    assert sparse["reference_rows"] == 1
    assert sparse["current_rows"] == 2
    assert sparse["reference_missing_fraction"] == 0.9
    assert sparse["current_missing_fraction"] == 0.8


def test_reference_prediction_features_rebuild_declared_duration():
    frame = pd.DataFrame(
        {
            "connectionTimeLocal": pd.to_datetime(["2020-01-01 08:00"]),
            "siteID": ["0001"],
            "clusterID": ["0001"],
            "stationID": ["0001-S001"],
            "kWhRequested": [10.0],
            "milesRequested": [20.0],
            "minutesAvailable": [120.0],
            "WhPerMile": [300.0],
            "declared_duration_minutes": [90.0],
            "kWhDelivered": [8.0],
        }
    )

    class ConstantModel:
        def predict(self, features):
            return np.full(len(features), 7.0)

    spec = FeatureSpec(
        categories={"siteID": [], "clusterID": [], "stationID": []},
        feature_names=[],
    )
    scoring = _prediction_frame(
        frame, ConstantModel(), spec, reference=True
    )

    assert scoring["declared_duration_minutes"].iat[0] == 90.0
    assert scoring["predicted_kwh"].iat[0] == 7.0
    assert scoring["siteID_code"].iat[0] == "0001"


def test_pushgateway_exports_metric_values_and_detect_state(monkeypatch):
    pushed = {}
    monkeypatch.setattr(
        "prometheus_client.push_to_gateway",
        lambda gateway, **kwargs: pushed.update(
            {"gateway": gateway, "metrics": list(kwargs["registry"].collect())}
        ),
    )
    reports = [
        {
            "batch_month": "2020-04",
            "batch_rows": 240,
            "drift_detected": True,
            "scores": [
                {
                    "column": "kWhDelivered",
                    "psi": 0.23,
                    "ks_statistic": 0.3,
                    "ks_pvalue": 0.001,
                    "js_distance": 0.16,
                    "psi_alert": True,
                    "reference_rows": 1000,
                    "current_rows": 240,
                    "reference_missing_fraction": 0.0,
                    "current_missing_fraction": 0.05,
                }
            ],
        }
    ]

    push_drift_metrics(reports, "raw")

    assert pushed["gateway"] == "http://127.0.0.1:9091"
    metrics = {
        sample.name: sample.value
        for family in pushed["metrics"]
        for sample in family.samples
        if sample.labels.get("column") == "kWhDelivered"
    }
    assert metrics["evcharge_drift_psi"] == 0.23
    assert metrics["evcharge_drift_ks_pvalue"] == 0.001
    assert metrics["evcharge_drift_detected"] == 1.0
    assert metrics["evcharge_drift_non_missing_rows"] == 240
    assert metrics["evcharge_drift_missing_fraction"] == 0.05
    global_metrics = {
        sample.name: sample.value
        for family in pushed["metrics"]
        for sample in family.samples
    }
    assert global_metrics["evcharge_drift_latest_alert"] == 1.0
    assert global_metrics["evcharge_drift_latest_batch_rows"] == 240


def test_score_new_months_includes_latest_for_alert_state(monkeypatch):
    scored = []
    pushed = []
    monkeypatch.setattr("evcharge.monitoring.drift.load_reference", lambda _s: "ref")
    monkeypatch.setattr(
        "evcharge.monitoring.drift.load_production_model",
        lambda _settings, _source: ("model", "spec", "name", "1"),
    )
    monkeypatch.setattr(
        "evcharge.orchestration.flows._monthly_partitions",
        lambda _settings, _source: {"2020-03": [], "2021-09": []},
    )
    monkeypatch.setattr(
        "evcharge.monitoring.drift.score_month",
        lambda _settings, _source, month, *_args: (
            scored.append(month) or {"batch_month": month}
        ),
    )
    monkeypatch.setattr(
        "evcharge.monitoring.drift.push_drift_metrics",
        lambda reports, source: pushed.append((reports, source)),
    )

    reports = score_new_months(object(), "raw", ["2020-03"])

    assert scored == ["2020-03", "2021-09"]
    assert reports == [{"batch_month": "2020-03"}, {"batch_month": "2021-09"}]
    assert pushed == [(reports, "raw")]
