"""Load MLflow production model and run inference."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

from src.features.score import pm25_to_advisory, confidence_interval

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "smoke-forecaster"

# Model RMSE used for confidence intervals (updated on load)
_MODEL_RMSE = 8.0
_model = None
_model_version = "unknown"


def load_production_model():
    """Load the MLflow production model into memory."""
    global _model, _model_version, _MODEL_RMSE

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        _model = mlflow.pyfunc.load_model(model_uri)

        # Get version metadata
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if versions:
            v = versions[0]
            _model_version = v.version
            run = client.get_run(v.run_id)
            _MODEL_RMSE = run.data.metrics.get("rmse", 8.0)

        logger.info("Loaded production model v%s (RMSE=%.3f)", _model_version, _MODEL_RMSE)

    except Exception as exc:
        logger.warning("Could not load MLflow model: %s — using fallback", exc)
        _model = None
        _model_version = "fallback"


def _build_live_features(lat: float, lon: float) -> pd.DataFrame:
    """Fetch live weather + recent fire pressure for a single point."""
    from src.ingestion.weather import fetch_current_weather
    from src.ingestion.firms import fetch_firms
    from src.features.engineer import compute_fire_pressure

    weather = fetch_current_weather(lat, lon)
    now = datetime.utcnow()

    try:
        fires = fetch_firms(days=2)
    except Exception:
        fires = pd.DataFrame(columns=["latitude", "longitude", "frp"])

    fire_pressure = compute_fire_pressure(
        lat, lon, weather["wind_direction_10m"], fires
    )

    return pd.DataFrame(
        [
            {
                "pm25": 10.0,  # placeholder; live reading not available without OpenAQ
                "wind_speed_10m": weather["wind_speed_10m"],
                "wind_direction_10m": weather["wind_direction_10m"],
                "relative_humidity_2m": weather["relative_humidity_2m"],
                "pm25_lag_24h": 10.0,
                "pm25_lag_48h": 10.0,
                "hour": now.hour,
                "day_of_year": now.timetuple().tm_yday,
                "fire_pressure_score": fire_pressure,
            }
        ]
    )


def predict(lat: float, lon: float) -> dict:
    """Run inference for (lat, lon) using the production model.

    Returns:
        dict with advisory_score, risk_tier, pm25_estimate,
        confidence_lower, confidence_upper, model_version, color
    """
    global _model, _model_version, _MODEL_RMSE

    if _model is None:
        load_production_model()

    features = _build_live_features(lat, lon)

    if _model is not None:
        try:
            pm25_pred = float(_model.predict(features)[0])
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            pm25_pred = _fallback_prediction(features)
    else:
        pm25_pred = _fallback_prediction(features)

    pm25_pred = max(0.0, pm25_pred)
    advisory = pm25_to_advisory(pm25_pred)
    ci_lower, ci_upper = confidence_interval(pm25_pred, _MODEL_RMSE)

    return {
        "advisory_score": advisory["advisory_score"],
        "risk_tier": advisory["risk_tier"],
        "pm25_estimate": round(pm25_pred, 2),
        "confidence_lower": ci_lower,
        "confidence_upper": ci_upper,
        "model_version": str(_model_version),
        "color": advisory["color"],
    }


def _fallback_prediction(features: pd.DataFrame) -> float:
    """Simple fallback: weighted average of lag + fire pressure signal."""
    row = features.iloc[0]
    fire_signal = min(row["fire_pressure_score"] * 0.01, 30)
    return 10.0 + fire_signal
