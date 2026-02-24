"""FastAPI application: /predict, /health, /drift-report."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.serving.schemas import HealthResponse, PredictRequest, PredictResponse
from src.serving.predict import load_production_model, predict

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "data/reports/drift_latest.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_production_model()
    yield


app = FastAPI(
    title="Wildfire Smoke Advisory Forecaster",
    description=(
        "Predicts PM2.5 concentration 24h ahead for any location in North America "
        "using NASA FIRMS fire data, OpenAQ air quality, and Open-Meteo wind data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms service is running and model is loaded."""
    from src.serving.predict import _model, _model_version as mv
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=str(mv),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Predict PM2.5 and advisory score for a given lat/lon.

    Returns a smoke advisory score (0–100), risk tier, PM2.5 estimate, and
    confidence interval for 24h ahead.
    """
    try:
        result = predict(request.lat, request.lon)
        return PredictResponse(**result)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/drift-report", response_class=HTMLResponse)
async def drift_report():
    """Serve the latest Evidently data drift HTML report."""
    report_path = Path(DRIFT_REPORT_PATH)
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Drift report not found. Run src/monitoring/drift.py to generate it.",
        )
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=True)
