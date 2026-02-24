"""Fetch wind speed, wind direction, and humidity from Open-Meteo."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(
    locations: list[dict],
    days: int = 30,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly wind + humidity for a list of locations.

    Args:
        locations: List of dicts with name/lat/lon
        days: Number of past days (uses archive API for >2 days)
        output_path: Optional path to save raw CSV

    Returns:
        DataFrame with columns: city, lat, lon, datetime, wind_speed_10m,
        wind_direction_10m, relative_humidity_2m
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)

    all_rows = []
    for loc in locations:
        rows = _fetch_location(loc, str(start), str(end))
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No weather data — using synthetic")
        return _synthetic_weather(locations, days)

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved weather data to %s (%d rows)", output_path, len(df))

    return df


def _fetch_location(loc: dict, start: str, end: str) -> list[dict]:
    """Fetch hourly weather for a single location using archive API."""
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,wind_direction_10m,relative_humidity_2m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Weather fetch failed for %s: %s", loc["name"], exc)
        return []

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    wind_speed = hourly.get("wind_speed_10m", [None] * len(times))
    wind_dir = hourly.get("wind_direction_10m", [None] * len(times))
    humidity = hourly.get("relative_humidity_2m", [None] * len(times))

    rows = []
    for t, ws, wd, rh in zip(times, wind_speed, wind_dir, humidity):
        rows.append(
            {
                "city": loc["name"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "datetime": t,
                "wind_speed_10m": ws,
                "wind_direction_10m": wd,
                "relative_humidity_2m": rh,
            }
        )

    logger.info("  %s: %d hourly weather records", loc["name"], len(rows))
    return rows


def fetch_current_weather(lat: float, lon: float) -> dict:
    """Fetch current (live) weather for a single point — used by serving."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "wind_speed_10m,wind_direction_10m,relative_humidity_2m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        current = resp.json().get("current", {})
        return {
            "wind_speed_10m": current.get("wind_speed_10m", 3.0),
            "wind_direction_10m": current.get("wind_direction_10m", 180.0),
            "relative_humidity_2m": current.get("relative_humidity_2m", 50.0),
        }
    except Exception as exc:
        logger.warning("Live weather fetch failed: %s", exc)
        return {
            "wind_speed_10m": 3.0,
            "wind_direction_10m": 180.0,
            "relative_humidity_2m": 50.0,
        }


def _synthetic_weather(locations: list[dict], days: int) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(1)
    end = datetime.utcnow()
    rows = []
    for loc in locations:
        for h in range(days * 24):
            ts = end - timedelta(hours=days * 24 - h)
            rows.append(
                {
                    "city": loc["name"],
                    "lat": loc["lat"],
                    "lon": loc["lon"],
                    "datetime": ts,
                    "wind_speed_10m": abs(rng.normal(4, 2)),
                    "wind_direction_10m": rng.uniform(0, 360),
                    "relative_humidity_2m": rng.uniform(30, 90),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.ingestion.openaq import CITIES

    data = fetch_weather(CITIES[:3], days=7, output_path="data/raw/weather.csv")
    print(data.head())
    print(f"Shape: {data.shape}")
