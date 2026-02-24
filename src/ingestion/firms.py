"""Fetch NASA FIRMS active fire detections for North America."""

import os
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FIRMS_API_KEY = os.getenv("FIRMS_API_KEY", "")
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# North America bounding box: west, south, east, north
NA_BBOX = "-170,15,-50,75"
PRODUCT = "VIIRS_SNPP_NRT"


def fetch_firms(
    bbox: str = NA_BBOX,
    days: int = 10,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Fetch FIRMS fire detections and return as DataFrame.

    Args:
        bbox: Bounding box string "west,south,east,north"
        days: Number of days of data to fetch (max 10 for NRT)
        output_path: Optional path to save raw CSV

    Returns:
        DataFrame with columns: latitude, longitude, bright_ti4, scan, track,
        acq_date, acq_time, satellite, instrument, confidence, version,
        bright_ti5, frp, daynight
    """
    if not FIRMS_API_KEY:
        logger.warning("FIRMS_API_KEY not set — returning synthetic data")
        return _synthetic_firms(days)

    url = f"{BASE_URL}/{FIRMS_API_KEY}/{PRODUCT}/{bbox}/{days}"
    logger.info("Fetching FIRMS data from %s", url)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df = _clean_firms(df)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved FIRMS data to %s (%d rows)", output_path, len(df))

    return df


def _clean_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and parse timestamps."""
    df.columns = [c.lower() for c in df.columns]
    # Parse datetime
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        errors="coerce",
    )
    # Keep confidence as string category: low/nominal/high
    df["confidence"] = df["confidence"].astype(str).str.lower()
    # Numeric columns
    for col in ["latitude", "longitude", "frp", "bright_ti4"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "frp"])
    return df


def _synthetic_firms(days: int = 10) -> pd.DataFrame:
    """Generate synthetic fire data for testing without an API key."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    times = [start + timedelta(hours=int(h)) for h in rng.uniform(0, days * 24, n)]

    return pd.DataFrame(
        {
            "latitude": rng.uniform(40, 65, n),
            "longitude": rng.uniform(-130, -70, n),
            "frp": rng.exponential(50, n),
            "confidence": rng.choice(["low", "nominal", "high"], n),
            "bright_ti4": rng.uniform(300, 450, n),
            "acq_date": [t.strftime("%Y-%m-%d") for t in times],
            "acq_time": [t.strftime("%H%M") for t in times],
            "acq_datetime": times,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_firms(days=3, output_path="data/raw/firms.csv")
    print(data.head())
    print(f"Total fires: {len(data)}")
