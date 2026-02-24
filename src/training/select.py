"""Compare MLflow runs and promote the best model to production registry."""

from __future__ import annotations

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "wildfire-smoke-pm25"
REGISTERED_MODEL_NAME = "smoke-forecaster"


def get_best_run() -> dict:
    """Find the MLflow run with the lowest validation RMSE.

    Returns:
        dict with run_id, rmse, model_name
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in experiment")

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "rmse": best.data.metrics.get("rmse", float("inf")),
        "model_name": best.data.params.get("model_type", "unknown"),
    }


def get_production_rmse(client: MlflowClient) -> float | None:
    """Return the RMSE of the current production model, or None if no production version."""
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        prod_run_id = versions[0].run_id
        prod_run = client.get_run(prod_run_id)
        return prod_run.data.metrics.get("rmse")
    except Exception:
        return None


def promote_best_model(best_run: dict | None = None) -> str:
    """Register best run and promote to Production if it beats current model.

    Args:
        best_run: dict from get_best_run(); fetched automatically if None

    Returns:
        'promoted' | 'unchanged' | 'first_registration'
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if best_run is None:
        best_run = get_best_run()

    run_id = best_run["run_id"]
    new_rmse = best_run["rmse"]
    model_name = best_run["model_name"]

    logger.info("Best run: %s  RMSE=%.4f  (run_id=%s)", model_name, new_rmse, run_id)

    # Register the model (creates new version in registry)
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    version_number = model_version.version
    logger.info("Registered model version %s", version_number)

    # Transition to staging first
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version_number,
        stage="Staging",
        archive_existing_versions=False,
    )

    prod_rmse = get_production_rmse(client)

    if prod_rmse is None:
        # No production model yet — promote directly
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version_number,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("First model — promoted v%s to Production (RMSE=%.4f)", version_number, new_rmse)
        return "first_registration"

    if new_rmse < prod_rmse:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version_number,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Promoted v%s to Production: new RMSE=%.4f < old RMSE=%.4f",
            version_number, new_rmse, prod_rmse,
        )
        return "promoted"
    else:
        logger.info(
            "Keeping existing production model: new RMSE=%.4f >= old RMSE=%.4f",
            new_rmse, prod_rmse,
        )
        return "unchanged"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    outcome = promote_best_model()
    print(f"Promotion outcome: {outcome}")
