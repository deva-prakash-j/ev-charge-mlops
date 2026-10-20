"""FastAPI serving application for the saved EV charge-energy model."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict

from evcharge.config import PROJECT_ROOT, get_settings, load_config
from evcharge.features.build import FeatureSpec, transform
from evcharge.ingest.normalise import normalise_identifier
from evcharge.models.registry import resolve_tracking_uri

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    siteID: str
    connectionTimeLocal: datetime
    clusterID: str | None = None
    stationID: str | None = None
    connectionTime: datetime | None = None
    requestedDeparture: datetime | None = None
    kWhRequested: float | None = None
    milesRequested: float | None = None
    minutesAvailable: float | None = None
    WhPerMile: float | None = None


def request_to_features(
    payload: PredictionRequest | dict[str, Any] | list[dict[str, Any]],
    spec: FeatureSpec,
    feature_config: dict[str, Any],
    data_config: dict[str, Any],
) -> pd.DataFrame:
    """Apply request validation and ingest identifier normalization before encoding."""
    payloads = payload if isinstance(payload, list) else [payload]
    records = []
    for item in payloads:
        request = (
            item
            if isinstance(item, PredictionRequest)
            else PredictionRequest.model_validate(item)
        )
        record = request.model_dump()
        for column in ("siteID", "clusterID", "stationID"):
            record[column] = normalise_identifier(record.get(column))
        records.append(record)
    return transform(pd.DataFrame(records), spec, feature_config, data_config)


def create_app(
    source: str = "sample",
    models_dir: Path | None = None,
    registry_model_name: str | None = None,
    prediction_log_path: Path | None = None,
) -> FastAPI:
    if source not in {"sample", "raw"}:
        raise ValueError("source must be 'sample' or 'raw'")

    if registry_model_name == "":
        registry_model_name = None
    model_dir = (models_dir or PROJECT_ROOT / "models") / source
    model_config = load_config("model")
    tracking_uri = resolve_tracking_uri(get_settings(), model_config)
    metrics_registry = CollectorRegistry()
    prediction_counter = Counter(
        "evcharge_predictions_total",
        "Successful predictions served by the EV Charge API.",
        ("model_name", "model_version"),
        registry=metrics_registry,
    )
    prediction_errors = Counter(
        "evcharge_prediction_errors_total",
        "Prediction errors returned by the EV Charge API.",
        ("model_name", "model_version", "reason"),
        registry=metrics_registry,
    )
    prediction_duration = Histogram(
        "evcharge_prediction_duration_seconds",
        "Time spent building features and predicting.",
        ("model_name", "model_version"),
        registry=metrics_registry,
    )
    prediction_distribution = Histogram(
        "evcharge_prediction_kwh",
        "Distribution of predicted energy in kWh.",
        ("model_name", "model_version"),
        buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 50, 100),
        registry=metrics_registry,
    )
    prediction_log_lock = threading.Lock()

    def log_prediction(event: dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True, allow_nan=False)
        logger.info("%s", line)
        if prediction_log_path is None:
            return
        prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_log_lock:
            with prediction_log_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as log_file:
                log_file.write(line + "\n")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.feature_config = load_config("features")
        app.state.data_config = load_config("data")
        if registry_model_name is not None:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            try:
                version = client.get_model_version_by_alias(
                    registry_model_name, "production"
                )
            except MlflowException as exc:
                raise RuntimeError(
                    f"Cannot load Production alias for registered model "
                    f"{registry_model_name!r} from {tracking_uri}: {exc}"
                ) from exc
            feature_spec_artifact = version.tags.get("feature_spec_artifact")
            if not feature_spec_artifact:
                raise RuntimeError(
                    f"Production version {registry_model_name}/{version.version} "
                    "has no feature_spec_artifact tag."
                )
            spec_path = Path(
                client.download_artifacts(version.run_id, feature_spec_artifact)
            )
            app.state.model = mlflow.sklearn.load_model(
                f"models:/{registry_model_name}@production"
            )
            app.state.feature_spec = FeatureSpec.load(spec_path)
            app.state.model_source = version.tags.get("source", source)
            app.state.model_name = registry_model_name
            app.state.model_version = str(version.version)
        else:
            model_path = model_dir / "model.joblib"
            spec_path = model_dir / "feature_spec.json"
            if not model_path.is_file() or not spec_path.is_file():
                raise RuntimeError(
                    f"Model artifacts are missing from {model_dir}. "
                    f"Train with: python -m evcharge.pipeline --source {source}"
                )
            app.state.model = joblib.load(model_path)
            app.state.feature_spec = FeatureSpec.load(spec_path)
            app.state.model_source = source
            app.state.model_name = f"local-{source}-energy"
            app.state.model_version = "local"
        yield

    app = FastAPI(
        title="EV Charge-Session Energy Prediction",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        if not hasattr(request.app.state, "model"):
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        result = {
            "status": "ok",
            "source": request.app.state.model_source,
        }
        if registry_model_name is not None:
            result["model_name"] = request.app.state.model_name
            result["model_version"] = request.app.state.model_version
        return result

    @app.post("/predict")
    def predict(
        payload: PredictionRequest, request: Request
    ) -> dict[str, Any]:
        if not hasattr(request.app.state, "model"):
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        model_name = request.app.state.model_name
        model_version = request.app.state.model_version
        labels = {"model_name": model_name, "model_version": model_version}
        started = time.perf_counter()
        try:
            features = request_to_features(
                payload,
                request.app.state.feature_spec,
                request.app.state.feature_config,
                request.app.state.data_config,
            )
            prediction = float(request.app.state.model.predict(features)[0])
            if not np.isfinite(prediction):
                prediction_errors.labels(
                    **labels, reason="non_finite_prediction"
                ).inc()
                raise HTTPException(
                    status_code=500,
                    detail="Model produced a non-finite prediction.",
                )

            feature_values = {
                name: float(value) if np.isfinite(value) else None
                for name, value in features.iloc[0].items()
            }
            log_prediction(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model_name": model_name,
                    "model_version": model_version,
                    "predicted_kwh": prediction,
                    "features": feature_values,
                }
            )
            prediction_counter.labels(**labels).inc()
            prediction_distribution.labels(**labels).observe(prediction)
            return {
                "predicted_kWhDelivered": prediction,
                "model_name": model_name,
                "model_version": model_version,
            }
        finally:
            prediction_duration.labels(**labels).observe(
                time.perf_counter() - started
            )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(
            content=generate_latest(metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app(
    source=os.getenv("EVCHARGE_MODEL_SOURCE", "sample"),
    registry_model_name=os.getenv("EVCHARGE_MODEL_NAME"),
    prediction_log_path=(
        Path(os.environ["EVCHARGE_PREDICTION_LOG"])
        if os.getenv("EVCHARGE_PREDICTION_LOG")
        else None
    ),
)
