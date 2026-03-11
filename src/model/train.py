"""Model training with XGBoost + Optuna + MLflow tracking."""

import logging
from typing import Any

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from src.config import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    OPTUNA_N_TRIALS,
    RANDOM_STATE,
)
from src.model.evaluate import evaluate_model

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_xgboost_model(params: dict | None = None) -> xgb.XGBRegressor:
    """Create an XGBRegressor with given or default parameters."""
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = OPTUNA_N_TRIALS,
) -> dict[str, Any]:
    """Use Optuna to find optimal XGBoost hyperparameters."""
    logger.info("Starting Optuna optimization with %d trials...", n_trials)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)

        # 5-fold CV, scoring = neg_RMSE on log-transformed target
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        return scores.mean()  # Negative RMSE, Optuna maximizes by default

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0

    logger.info("Optuna best CV RMSE (log scale): %.5f", -study.best_value)
    logger.info("Best params: %s", best_params)
    return best_params


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    optimize: bool = True,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple[xgb.XGBRegressor, dict]:
    """Train XGBoost model with optional Optuna tuning and MLflow tracking.

    Args:
        X_train, y_train: Training data (y is log-transformed).
        X_test, y_test: Test data (y is log-transformed).
        feature_names: List of feature column names.
        optimize: Whether to run Optuna hyperparameter optimization.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    # Derive feature_names from training data if not provided
    if feature_names is None:
        feature_names = list(X_train.columns)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logger.info("MLflow run started: %s", run.info.run_id)

        # Hyperparameter optimization
        if optimize:
            best_params = optimize_hyperparameters(X_train, y_train, n_trials)
            model = create_xgboost_model(best_params)
            mlflow.log_params(best_params)
        else:
            model = create_xgboost_model()
            mlflow.log_params(model.get_params())

        # Train
        logger.info("Training XGBoost model...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate (on original price scale)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log feature importance
        importance = dict(zip(X_train.columns, model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        for feat, imp in sorted_importance.items():
            mlflow.log_metric(f"importance_{feat}", imp)

        # Log dataset info
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", list(X_train.columns))

        # Log model to MLflow registry
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="house-price-xgboost",
            input_example=X_train.iloc[:1],
        )

        # Encoder and feature_names are stored on MinIO S3 by the data pipeline.
        # No need to duplicate them as MLflow artifacts.

        logger.info("Model logged to MLflow. Run ID: %s", run.info.run_id)
        logger.info("Metrics: %s", metrics)

        return model, metrics


def train_pipeline(
    data: dict,
    optimize: bool = True,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple[xgb.XGBRegressor, dict]:
    """End-to-end training pipeline.

    Args:
        data: Output of run_preprocessing_pipeline().
        optimize: Whether to run Optuna.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    return train_model(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        feature_names=data.get("feature_names"),
        optimize=optimize,
        n_trials=n_trials,
    )
