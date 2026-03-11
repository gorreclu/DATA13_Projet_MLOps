"""Model evaluation utilities."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocess import inverse_transform_target

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model on test set.

    Computes metrics on both log-transformed and original price scales.

    Args:
        model: Trained model with .predict().
        X_test: Test features.
        y_test: Test target (log-transformed).

    Returns:
        Dict of metric name -> value.
    """
    y_pred_log = model.predict(X_test)

    # Metrics on log scale
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)

    # Metrics on original price scale
    y_test_real = inverse_transform_target(y_test.values)
    y_pred_real = inverse_transform_target(y_pred_log)

    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    r2_real = r2_score(y_test_real, y_pred_real)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

    metrics = {
        "rmse_log": round(rmse_log, 5),
        "mae_log": round(mae_log, 5),
        "r2_log": round(r2_log, 5),
        "rmse_price": round(rmse_real, 2),
        "mae_price": round(mae_real, 2),
        "r2_price": round(r2_real, 5),
        "mape": round(mape, 2),
    }

    logger.info("Evaluation results:")
    for name, value in metrics.items():
        logger.info("  %s: %s", name, value)

    return metrics


def compare_models(
    current_metrics: dict[str, float],
    previous_metrics: dict[str, float] | None,
    primary_metric: str = "rmse_price",
    lower_is_better: bool = True,
) -> bool:
    """Compare current model against previous model.

    Returns True if the current model is better.
    """
    if previous_metrics is None:
        logger.info("No previous model to compare. Current model is best.")
        return True

    current_val = current_metrics.get(primary_metric)
    previous_val = previous_metrics.get(primary_metric)

    if current_val is None or previous_val is None:
        logger.warning("Metric '%s' not found. Defaulting to current model.", primary_metric)
        return True

    if lower_is_better:
        is_better = current_val < previous_val
    else:
        is_better = current_val > previous_val

    logger.info(
        "Model comparison on '%s': current=%.4f, previous=%.4f -> %s",
        primary_metric,
        current_val,
        previous_val,
        "BETTER" if is_better else "WORSE",
    )
    return is_better
