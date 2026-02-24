"""Train Ridge / RF / XGBoost / LightGBM; log each run to MLflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "wildfire-smoke-pm25"

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
TARGET = "pm25_next_24h"


def _get_models() -> list[tuple[str, object]]:
    """Return list of (name, unfitted model) pairs."""
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        xgb_model = None

    try:
        import lightgbm as lgb
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        lgbm_model = None

    models = [
        ("Ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ]
    if xgb_model:
        models.append(("XGBoost", xgb_model))
    if lgbm_model:
        models.append(("LightGBM", lgbm_model))

    return models


def _log_feature_importance(model, feature_names: list[str], model_name: str) -> None:
    """Save and log a feature importance bar chart to MLflow."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps"):
            inner = model.named_steps.get("model")
            if inner and hasattr(inner, "coef_"):
                importances = np.abs(inner.coef_)
            else:
                return
        else:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        idx = np.argsort(importances)
        ax.barh([feature_names[i] for i in idx], importances[idx])
        ax.set_title(f"{model_name} — Feature Importance")
        ax.set_xlabel("Importance")
        plt.tight_layout()

        img_path = f"/tmp/{model_name}_importance.png"
        fig.savefig(img_path)
        plt.close(fig)
        mlflow.log_artifact(img_path, artifact_path="plots")
    except Exception as exc:
        logger.warning("Could not log feature importance: %s", exc)


def train_all(
    features_path: str = "data/features/features.csv",
    test_size: float = 0.2,
) -> list[dict]:
    """Train all candidate models and log to MLflow.

    Returns:
        List of dicts with keys: run_id, model_name, rmse, mae, r2
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(features_path)
    X = df[FEATURE_COLS].values
    y = df[TARGET].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    results = []
    for name, model in _get_models():
        logger.info("Training %s ...", name)
        with mlflow.start_run(run_name=name) as run:
            # Log hyperparams
            mlflow.log_param("model_type", name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_val", len(X_val))
            mlflow.log_param("features", FEATURE_COLS)

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            mae = float(mean_absolute_error(y_val, preds))
            r2 = float(r2_score(y_val, preds))

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log model artifact
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            # Feature importance plot
            _log_feature_importance(model, FEATURE_COLS, name)

            logger.info("  %s → RMSE=%.3f  MAE=%.3f  R²=%.3f", name, rmse, mae, r2)
            results.append(
                {
                    "run_id": run.info.run_id,
                    "model_name": name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = train_all()
    for r in sorted(res, key=lambda x: x["rmse"]):
        print(f"{r['model_name']:15s}  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}  R²={r['r2']:.3f}")
