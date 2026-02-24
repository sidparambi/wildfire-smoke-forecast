"""Pydantic request/response models for the FastAPI serving layer."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")

    model_config = {"json_schema_extra": {"example": {"lat": 43.65, "lon": -79.38}}}


class PredictResponse(BaseModel):
    advisory_score: int = Field(..., ge=0, le=100, description="0–100 smoke advisory score")
    risk_tier: str = Field(..., description="Low | Moderate | High | Hazardous")
    pm25_estimate: float = Field(..., ge=0, description="Predicted PM2.5 μg/m³ (24h ahead)")
    confidence_lower: float = Field(..., ge=0, description="Lower bound of 95% CI")
    confidence_upper: float = Field(..., ge=0, description="Upper bound of 95% CI")
    model_version: str = Field(..., description="MLflow model version tag")
    color: str = Field(..., description="Risk color: green | yellow | orange | red")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "advisory_score": 42,
                "risk_tier": "Moderate",
                "pm25_estimate": 22.5,
                "confidence_lower": 14.5,
                "confidence_upper": 30.5,
                "model_version": "2",
                "color": "yellow",
            }
        },
    }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None

    model_config = {"protected_namespaces": ()}
