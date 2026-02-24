"""Prefect flow: ingest → features → train → evaluate → promote."""

from __future__ import annotations

import logging
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.schedules import CronSchedule

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
FEATURES_PATH = "data/features/features.csv"


@task(name="fetch-firms", retries=2, retry_delay_seconds=60)
def fetch_firms_task(days: int = 10) -> str:
    from src.ingestion.firms import fetch_firms

    log = get_run_logger()
    output = str(RAW_DIR / "firms.csv")
    df = fetch_firms(days=days, output_path=output)
    log.info("FIRMS: %d fire detections", len(df))
    return output


@task(name="fetch-openaq", retries=2, retry_delay_seconds=60)
def fetch_openaq_task(days: int = 30) -> str:
    from src.ingestion.openaq import fetch_openaq

    log = get_run_logger()
    output = str(RAW_DIR / "openaq.csv")
    df = fetch_openaq(days=days, output_path=output)
    log.info("OpenAQ: %d PM2.5 readings", len(df))
    return output


@task(name="fetch-weather", retries=2, retry_delay_seconds=60)
def fetch_weather_task(days: int = 30) -> str:
    from src.ingestion.openaq import CITIES
    from src.ingestion.weather import fetch_weather

    log = get_run_logger()
    output = str(RAW_DIR / "weather.csv")
    df = fetch_weather(CITIES, days=days, output_path=output)
    log.info("Weather: %d hourly records", len(df))
    return output


@task(name="engineer-features")
def engineer_features_task(
    openaq_path: str, firms_path: str, weather_path: str
) -> str:
    import pandas as pd
    from src.features.engineer import build_features

    log = get_run_logger()
    aq = pd.read_csv(openaq_path, parse_dates=["datetime"])
    firms = pd.read_csv(firms_path, parse_dates=["acq_datetime"])
    wx = pd.read_csv(weather_path, parse_dates=["datetime"])

    df = build_features(aq, firms, wx, output_path=FEATURES_PATH)
    log.info("Feature table: %s rows × %s cols", *df.shape)
    return FEATURES_PATH


@task(name="train-models")
def train_task(features_path: str) -> list[dict]:
    from src.training.train import train_all

    log = get_run_logger()
    results = train_all(features_path=features_path)
    for r in results:
        log.info("  %s  RMSE=%.3f", r["model_name"], r["rmse"])
    return results


@task(name="promote-best-model")
def promote_task(results: list[dict]) -> str:
    from src.training.select import promote_best_model, get_best_run

    log = get_run_logger()
    best = min(results, key=lambda x: x["rmse"])
    log.info("Best run: %s (RMSE=%.3f)", best["model_name"], best["rmse"])
    outcome = promote_best_model(best_run=best)
    log.info("Promotion outcome: %s", outcome)
    return outcome


@task(name="generate-drift-report")
def drift_task() -> str:
    from src.monitoring.drift import generate_drift_report

    log = get_run_logger()
    path = generate_drift_report()
    log.info("Drift report: %s", path)
    return path


@flow(
    name="wildfire-smoke-pipeline",
    description="End-to-end pipeline: data ingestion → feature engineering → model training → promotion",
)
def run_pipeline(ingest_days: int = 30, retrain: bool = True) -> dict:
    """Full MLOps pipeline.

    Args:
        ingest_days: Days of historical data to fetch
        retrain: Whether to train new models (set False for inference-only runs)
    """
    # Parallel ingestion
    firms_path = fetch_firms_task(days=min(ingest_days, 10))
    aq_path = fetch_openaq_task(days=ingest_days)
    wx_path = fetch_weather_task(days=ingest_days)

    # Sequential: engineer features → train → promote
    features_path = engineer_features_task(aq_path, firms_path, wx_path)

    outcome = "skipped"
    if retrain:
        results = train_task(features_path)
        outcome = promote_task(results)

    drift_path = drift_task()

    return {
        "features_path": features_path,
        "promotion_outcome": outcome,
        "drift_report": drift_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline(ingest_days=7)
    print(result)
