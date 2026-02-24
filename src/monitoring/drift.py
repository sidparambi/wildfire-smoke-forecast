"""Generate Evidently data drift report comparing training baseline vs. recent inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "data/reports/drift_latest.html")
FEATURES_PATH = "data/features/features.csv"
INFERENCE_LOG_PATH = "data/predictions/inference_log.csv"

FEATURE_COLS = [
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


def generate_drift_report(
    reference_path: str = FEATURES_PATH,
    current_path: str = INFERENCE_LOG_PATH,
    output_path: str = REPORT_PATH,
    lookback_days: int = 7,
) -> str:
    """Generate an Evidently DataDriftPreset report.

    Args:
        reference_path: Path to training feature CSV (baseline)
        current_path: Path to inference log CSV (recent predictions)
        output_path: Where to save the HTML report
        lookback_days: How many days of inference data to include

    Returns:
        Path to the generated HTML report
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.error("evidently not installed — run: pip install evidently")
        raise

    # Load reference (training) data
    if not Path(reference_path).exists():
        raise FileNotFoundError(f"Reference data not found: {reference_path}")

    ref_df = pd.read_csv(reference_path)[FEATURE_COLS].dropna()

    # Load or synthesize current (recent inference) data
    if Path(current_path).exists():
        cur_df = pd.read_csv(current_path)
        if "timestamp" in cur_df.columns:
            cur_df["timestamp"] = pd.to_datetime(cur_df["timestamp"])
            cutoff = cur_df["timestamp"].max() - pd.Timedelta(days=lookback_days)
            cur_df = cur_df[cur_df["timestamp"] >= cutoff]
        cur_df = cur_df[[c for c in FEATURE_COLS if c in cur_df.columns]].dropna()
    else:
        logger.warning("Inference log not found — using random sample of reference as current")
        cur_df = ref_df.sample(min(200, len(ref_df)), random_state=99).copy()
        # Add slight noise to simulate drift
        import numpy as np
        rng = np.random.default_rng(99)
        for col in ["pm25", "fire_pressure_score"]:
            if col in cur_df.columns:
                cur_df[col] = cur_df[col] * rng.uniform(1.05, 1.3, len(cur_df))

    if len(cur_df) < 10:
        logger.warning("Too few current records (%d) — drift report may be unreliable", len(cur_df))

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to %s", output_path)
    return output_path


def log_inference(features: dict, pm25_pred: float, timestamp: str | None = None) -> None:
    """Append a single inference record to the inference log CSV."""
    import datetime

    if timestamp is None:
        timestamp = datetime.datetime.utcnow().isoformat()

    row = {**features, "pm25_pred": pm25_pred, "timestamp": timestamp}
    log_path = Path(INFERENCE_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_drift_report()
    print(f"Report saved to: {path}")
