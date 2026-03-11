"""Model loading and prediction utilities."""

import logging

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MLFLOW_TRACKING_URI
from src.data.preprocess import inverse_transform_target

logger = logging.getLogger(__name__)

# Cache the loaded model to avoid reloading on every request
_cached_model: xgb.XGBRegressor | None = None


def load_model_from_mlflow(
    model_name: str = "house-price-xgboost",
    stage: str = "latest",
) -> xgb.XGBRegressor:
    """Load the latest model from MLflow Model Registry.

    Args:
        model_name: Registered model name.
        stage: Model version stage ("latest", "staging", "production").

    Returns:
        Loaded XGBRegressor model.
    """
    global _cached_model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        if stage == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{stage}"

        logger.info("Loading model from MLflow: %s", model_uri)
        model = mlflow.xgboost.load_model(model_uri)
        _cached_model = model
        logger.info("Model loaded successfully.")
        return model

    except Exception as e:
        logger.error("Failed to load model from MLflow: %s", e)
        raise


def get_model(model_name: str = "house-price-xgboost") -> xgb.XGBRegressor:
    """Get the cached model or load it from MLflow."""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model_from_mlflow(model_name)
    return _cached_model


def predict(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict house prices (returns real-scale prices, not log).

    Args:
        model: Trained XGBRegressor.
        X: Preprocessed features DataFrame.

    Returns:
        Array of predicted prices in USD.
    """
    y_pred_log = model.predict(X)
    y_pred_real = inverse_transform_target(y_pred_log)
    return y_pred_real


def predict_single(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
) -> float:
    """Predict a single house price.

    Args:
        model: Trained XGBRegressor.
        X: Single row preprocessed DataFrame.

    Returns:
        Predicted price in USD.
    """
    prices = predict(model, X)
    return float(prices[0])
