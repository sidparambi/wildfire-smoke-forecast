"""Fetch PM2.5 ground sensor readings from OpenAQ v2 API."""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.openaq.org/v2/measurements"

# Canadian + US cities with known PM2.5 sensors
CITIES = [
    {"name": "Toronto", "lat": 43.65, "lon": -79.38},
    {"name": "Vancouver", "lat": 49.25, "lon": -123.12},
    {"name": "Calgary", "lat": 51.05, "lon": -114.07},
    {"name": "Edmonton", "lat": 53.55, "lon": -113.50},
    {"name": "Montreal", "lat": 45.50, "lon": -73.57},
    {"name": "Los Angeles", "lat": 34.05, "lon": -118.24},
    {"name": "Seattle", "lat": 47.61, "lon": -122.33},
    {"name": "Portland", "lat": 45.52, "lon": -122.68},
    {"name": "San Francisco", "lat": 37.77, "lon": -122.42},
    {"name": "Denver", "lat": 39.74, "lon": -104.98},
    {"name": "New York", "lat": 40.71, "lon": -74.01},
]


def fetch_openaq(
    days: int = 30,
    cities: list[dict] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Fetch PM2.5 readings for all cities over the past `days` days.

    Args:
        days: Number of past days to fetch
        cities: List of dicts with name/lat/lon; defaults to CITIES
        output_path: Optional path to save raw CSV

    Returns:
        DataFrame with columns: city, lat, lon, datetime, pm25
    """
    if cities is None:
        cities = CITIES

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    all_rows = []
    for city in cities:
        rows = _fetch_city(city, start, end)
        all_rows.extend(rows)
        time.sleep(0.3)  # rate-limit courtesy

    if not all_rows:
        logger.warning("No OpenAQ data returned — using synthetic data")
        return _synthetic_openaq(cities, days)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved OpenAQ data to %s (%d rows)", output_path, len(df))

    return df


def _fetch_city(city: dict, start: datetime, end: datetime) -> list[dict]:
    """Fetch PM2.5 readings for a single city location."""
    params = {
        "parameter": "pm25",
        "coordinates": f"{city['lat']},{city['lon']}",
        "radius": 25000,  # 25km radius
        "date_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 1000,
        "sort": "asc",
        "order_by": "datetime",
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as exc:
        logger.warning("OpenAQ fetch failed for %s: %s", city["name"], exc)
        return []

    rows = []
    for r in results:
        try:
            rows.append(
                {
                    "city": city["name"],
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "datetime": pd.to_datetime(r["date"]["utc"]),
                    "pm25": float(r["value"]),
                }
            )
        except (KeyError, ValueError):
            continue

    logger.info("  %s: %d PM2.5 readings", city["name"], len(rows))
    return rows


def _synthetic_openaq(cities: list[dict], days: int) -> pd.DataFrame:
    """Generate synthetic PM2.5 time series for testing."""
    import numpy as np

    rng = np.random.default_rng(0)
    end = datetime.utcnow()
    rows = []
    for city in cities:
        base_pm25 = rng.uniform(5, 25)
        for h in range(days * 24):
            ts = end - timedelta(hours=days * 24 - h)
            pm25 = max(0, base_pm25 + rng.normal(0, 5) + 10 * np.sin(h / 12 * np.pi))
            rows.append(
                {
                    "city": city["name"],
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "datetime": ts,
                    "pm25": round(pm25, 2),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_openaq(days=7, output_path="data/raw/openaq.csv")
    print(data.head())
    print(f"Shape: {data.shape}")
