"""Unit tests for feature engineering and score mapping."""

import math

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    bearing_deg,
    compute_fire_pressure,
    haversine_km,
    angular_diff,
    build_features,
    MODEL_FEATURES,
)
from src.features.score import pm25_to_advisory, confidence_interval


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def test_haversine_known_distance():
    # Toronto → Vancouver ≈ 3,346 km
    dist = haversine_km(43.65, -79.38, 49.25, -123.12)
    assert 3200 < dist < 3500


def test_haversine_same_point():
    assert haversine_km(45.0, -75.0, 45.0, -75.0) == 0.0


def test_bearing_north():
    # Point directly north
    b = bearing_deg(0, 0, 1, 0)
    assert abs(b - 0) < 1 or abs(b - 360) < 1


def test_bearing_east():
    b = bearing_deg(0, 0, 0, 1)
    assert abs(b - 90) < 1


def test_angular_diff_simple():
    assert angular_diff(10, 350) == pytest.approx(20, abs=0.1)
    assert angular_diff(0, 180) == pytest.approx(180, abs=0.1)
    assert angular_diff(90, 90) == pytest.approx(0, abs=0.1)


# ─── Fire pressure score ──────────────────────────────────────────────────────

def _make_fires(lat, lon, frp=100.0):
    return pd.DataFrame({"latitude": [lat], "longitude": [lon], "frp": [frp]})


def test_fire_pressure_empty():
    score = compute_fire_pressure(43.65, -79.38, 270, pd.DataFrame(columns=["latitude", "longitude", "frp"]))
    assert score == 0.0


def test_fire_pressure_upwind_scores_higher_than_downwind():
    sensor_lat, sensor_lon = 43.65, -79.38
    # Wind FROM west (270°) → smoke blows east → fire to east is upwind
    wind_dir = 270  # wind FROM west
    fire_east = _make_fires(sensor_lat, sensor_lon + 1.0)  # fire east of sensor
    fire_west = _make_fires(sensor_lat, sensor_lon - 1.0)  # fire west of sensor
    score_upwind = compute_fire_pressure(sensor_lat, sensor_lon, wind_dir, fire_east)
    score_downwind = compute_fire_pressure(sensor_lat, sensor_lon, wind_dir, fire_west)
    assert score_upwind > score_downwind


def test_fire_pressure_beyond_radius():
    # Fire 1000km away — beyond 500km radius
    fires = _make_fires(50.0, -20.0, frp=1000)
    score = compute_fire_pressure(43.65, -79.38, 0, fires)
    assert score == 0.0


def test_fire_pressure_scales_with_frp():
    sensor_lat, sensor_lon = 43.65, -79.38
    wind_dir = 270
    fires_low = _make_fires(sensor_lat, sensor_lon + 1.0, frp=10)
    fires_high = _make_fires(sensor_lat, sensor_lon + 1.0, frp=100)
    score_low = compute_fire_pressure(sensor_lat, sensor_lon, wind_dir, fires_low)
    score_high = compute_fire_pressure(sensor_lat, sensor_lon, wind_dir, fires_high)
    assert score_high == pytest.approx(score_low * 10, rel=0.01)


# ─── Advisory score mapping ───────────────────────────────────────────────────

@pytest.mark.parametrize("pm25,expected_tier,expected_color", [
    (0.0, "Low", "green"),
    (6.0, "Low", "green"),
    (12.0, "Low", "green"),
    (20.0, "Moderate", "yellow"),
    (35.4, "Moderate", "yellow"),
    (45.0, "High", "orange"),
    (60.0, "Hazardous", "red"),
    (200.0, "Hazardous", "red"),
])
def test_pm25_to_advisory_tiers(pm25, expected_tier, expected_color):
    result = pm25_to_advisory(pm25)
    assert result["risk_tier"] == expected_tier
    assert result["color"] == expected_color
    assert 0 <= result["advisory_score"] <= 100


def test_pm25_negative_clamped():
    result = pm25_to_advisory(-5.0)
    assert result["advisory_score"] == 0
    assert result["risk_tier"] == "Low"


def test_advisory_score_monotonic():
    scores = [pm25_to_advisory(pm25)["advisory_score"] for pm25 in [0, 5, 12, 20, 35, 50, 80]]
    assert scores == sorted(scores)


def test_confidence_interval_positive():
    lo, hi = confidence_interval(20.0, model_rmse=8.0)
    assert lo >= 0
    assert hi > lo
    assert hi == pytest.approx(28.0, abs=0.1)


# ─── build_features integration ──────────────────────────────────────────────

def _make_openaq(n_hours: int = 72, city: str = "Toronto") -> pd.DataFrame:
    base = pd.Timestamp("2024-07-01")
    hours = [base + pd.Timedelta(hours=h) for h in range(n_hours)]
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "city": city,
        "lat": 43.65,
        "lon": -79.38,
        "datetime": hours,
        "pm25": rng.uniform(5, 30, n_hours),
    })


def _make_weather(n_hours: int = 72, city: str = "Toronto") -> pd.DataFrame:
    base = pd.Timestamp("2024-07-01")
    hours = [base + pd.Timedelta(hours=h) for h in range(n_hours)]
    rng = np.random.default_rng(8)
    return pd.DataFrame({
        "city": city,
        "lat": 43.65,
        "lon": -79.38,
        "datetime": hours,
        "wind_speed_10m": rng.uniform(0, 10, n_hours),
        "wind_direction_10m": rng.uniform(0, 360, n_hours),
        "relative_humidity_2m": rng.uniform(30, 90, n_hours),
    })


def test_build_features_columns():
    aq = _make_openaq(100)
    wx = _make_weather(100)
    firms = pd.DataFrame(columns=["latitude", "longitude", "frp", "acq_datetime"])
    result = build_features(aq, firms, wx)
    for col in MODEL_FEATURES:
        assert col in result.columns, f"Missing column: {col}"


def test_build_features_no_nulls_in_model_cols():
    aq = _make_openaq(100)
    wx = _make_weather(100)
    firms = pd.DataFrame(columns=["latitude", "longitude", "frp", "acq_datetime"])
    result = build_features(aq, firms, wx)
    assert result[MODEL_FEATURES].isnull().sum().sum() == 0


def test_build_features_target_present():
    aq = _make_openaq(100)
    wx = _make_weather(100)
    firms = pd.DataFrame(columns=["latitude", "longitude", "frp", "acq_datetime"])
    result = build_features(aq, firms, wx)
    assert "pm25_next_24h" in result.columns
    assert len(result) > 0
