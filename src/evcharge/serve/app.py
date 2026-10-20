"""FastAPI serving application for the saved EV charge-energy model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from evcharge.config import PROJECT_ROOT, load_config
from evcharge.features.build import FeatureSpec, transform


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


def create_app(
    source: str = "sample", models_dir: Path | None = None
) -> FastAPI:
    if source not in {"sample", "raw"}:
        raise ValueError("source must be 'sample' or 'raw'")

    model_dir = (models_dir or PROJECT_ROOT / "models") / source

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model_path = model_dir / "model.joblib"
        spec_path = model_dir / "feature_spec.json"
        if not model_path.is_file() or not spec_path.is_file():
            raise RuntimeError(
                f"Model artifacts are missing from {model_dir}. "
                f"Train with: python -m evcharge.pipeline --source {source}"
            )
        app.state.model = joblib.load(model_path)
        app.state.feature_spec = FeatureSpec.load(spec_path)
        app.state.feature_config = load_config("features")
        app.state.data_config = load_config("data")
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
        return {"status": "ok", "source": source}

    @app.post("/predict")
    def predict(
        payload: PredictionRequest, request: Request
    ) -> dict[str, Any]:
        if not hasattr(request.app.state, "model"):
            raise HTTPException(status_code=503, detail="Model is not loaded.")

        frame = pd.DataFrame([payload.model_dump()])
        features = transform(
            frame,
            request.app.state.feature_spec,
            request.app.state.feature_config,
            request.app.state.data_config,
        )
        prediction = float(request.app.state.model.predict(features)[0])
        if not np.isfinite(prediction):
            raise HTTPException(
                status_code=500, detail="Model produced a non-finite prediction."
            )
        return {"predicted_kWhDelivered": prediction}

    return app


app = create_app(source=os.getenv("EVCHARGE_MODEL_SOURCE", "sample"))
