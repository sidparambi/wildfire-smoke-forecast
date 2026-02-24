"""Feature engineering: fire_pressure_score + lag features."""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIRE_INFLUENCE_RADIUS_KM = 500
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    r = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees [0, 360) from point 1 → point 2."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angular_diff(a: float, b: float) -> float:
    """Absolute angular difference between two bearings [0, 180]."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def compute_fire_pressure(
    sensor_lat: float,
    sensor_lon: float,
    wind_dir: float,
    fires_df: pd.DataFrame,
    radius_km: float = FIRE_INFLUENCE_RADIUS_KM,
) -> float:
    """Compute fire pressure score for a single sensor location.

    Upwind fires weighted by FRP / distance².
    Angular tolerance: fires within ±90° upwind direction contribute.

    Args:
        sensor_lat: Sensor latitude
        sensor_lon: Sensor longitude
        wind_dir: Wind direction in degrees (direction wind is blowing FROM)
        fires_df: DataFrame with latitude, longitude, frp columns
        radius_km: Search radius in km

    Returns:
        Scalar fire pressure score (≥ 0)
    """
    if fires_df.empty:
        return 0.0

    score = 0.0
    # Wind blows FROM wind_dir, so smoke travels TO (wind_dir + 180) % 360
    smoke_direction = (wind_dir + 180) % 360

    for _, fire in fires_df.iterrows():
        dist = haversine_km(sensor_lat, sensor_lon, fire["latitude"], fire["longitude"])
        if dist < 1 or dist > radius_km:
            continue

        fire_bearing = bearing_deg(sensor_lat, sensor_lon, fire["latitude"], fire["longitude"])
        # Angular difference between smoke travel direction and bearing to fire
        angle_diff = angular_diff(smoke_direction, fire_bearing)

        if angle_diff <= 90:
            # Cosine weighting: 1.0 at 0° offset, 0.0 at 90°
            upwind_weight = math.cos(math.radians(angle_diff))
            score += (fire["frp"] * upwind_weight) / (dist**2)

    return score


def build_features(
    openaq_df: pd.DataFrame,
    firms_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Join all data sources and engineer features.

    Returns one row per (city, hourly timestamp) with all model features
    plus target column `pm25_next_24h`.
    """
    logger.info("Building features...")

    # --- Merge weather onto OpenAQ (nearest hour) ---
    aq = openaq_df.copy()
    wx = weather_df.copy()
    aq["datetime"] = pd.to_datetime(aq["datetime"]).dt.floor("h")
    wx["datetime"] = pd.to_datetime(wx["datetime"]).dt.floor("h")

    merged = aq.merge(
        wx[["city", "datetime", "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m"]],
        on=["city", "datetime"],
        how="left",
    )

    # Fill missing weather with reasonable defaults
    merged["wind_speed_10m"] = merged["wind_speed_10m"].fillna(3.0)
    merged["wind_direction_10m"] = merged["wind_direction_10m"].fillna(180.0)
    merged["relative_humidity_2m"] = merged["relative_humidity_2m"].fillna(50.0)

    # --- Lag features (per city) ---
    merged = merged.sort_values(["city", "datetime"])
    merged["pm25_lag_24h"] = merged.groupby("city")["pm25"].shift(24)
    merged["pm25_lag_48h"] = merged.groupby("city")["pm25"].shift(48)

    # --- Target: PM2.5 24h ahead ---
    merged["pm25_next_24h"] = merged.groupby("city")["pm25"].shift(-24)

    # --- Time features ---
    merged["hour"] = merged["datetime"].dt.hour
    merged["day_of_year"] = merged["datetime"].dt.dayofyear

    # --- Fire pressure score ---
    if firms_df.empty:
        merged["fire_pressure_score"] = 0.0
    else:
        firms_df = firms_df.copy()
        firms_df["acq_datetime"] = pd.to_datetime(firms_df["acq_datetime"])

        scores = []
        for _, row in merged.iterrows():
            # Fires within 48h prior to this observation
            cutoff = row["datetime"]
            window_start = cutoff - pd.Timedelta(hours=48)
            recent_fires = firms_df[
                (firms_df["acq_datetime"] >= window_start)
                & (firms_df["acq_datetime"] <= cutoff)
            ]
            score = compute_fire_pressure(
                row["lat"], row["lon"], row["wind_direction_10m"], recent_fires
            )
            scores.append(score)
        merged["fire_pressure_score"] = scores

    feature_cols = [
        "city",
        "lat",
        "lon",
        "datetime",
        "pm25",
        "wind_speed_10m",
        "wind_direction_10m",
        "relative_humidity_2m",
        "pm25_lag_24h",
        "pm25_lag_48h",
        "hour",
        "day_of_year",
        "fire_pressure_score",
        "pm25_next_24h",
    ]
    result = merged[feature_cols].dropna(subset=["pm25_lag_24h", "pm25_lag_48h", "pm25_next_24h"])
    result = result.reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("Saved feature table to %s (%d rows)", output_path, len(result))

    logger.info("Feature table shape: %s", result.shape)
    return result


MODEL_FEATURES = [
    "pm25",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "pm25_lag_24h",
    "pm25_lag_48h",
    "hour",
    "day_of_year",
    "fire_pressure_score",
]
TARGET = "pm25_next_24h"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.ingestion.firms import fetch_firms
    from src.ingestion.openaq import fetch_openaq, CITIES
    from src.ingestion.weather import fetch_weather

    firms = fetch_firms(days=3)
    aq = fetch_openaq(days=7, cities=CITIES[:3])
    wx = fetch_weather(CITIES[:3], days=7)
    feats = build_features(aq, firms, wx, output_path="data/features/features.csv")
    print(feats.head())
