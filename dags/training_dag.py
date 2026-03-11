"""Airflow DAG: Full training pipeline reading preprocessed data from MinIO S3.

Runs on demand (no schedule) or can be triggered by the data pipeline DAG.
Reads preprocessed data from MinIO, trains XGBoost with Optuna, logs to MLflow.
Compares the new model against the previous one and only promotes it if better.
Sends email alerts on success/failure via Gmail SMTP.
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# S3 keys matching those written by data_pipeline DAG
S3_PROCESSED_PREFIX = "processed/"

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _ensure_data_on_minio(**kwargs):
    """Check that preprocessed data exists on MinIO.

    If not, fall back to downloading from Kaggle, preprocessing, and uploading.
    This makes the training DAG self-sufficient even when triggered manually.
    """
    from src.data.s3_storage import file_exists

    required_keys = [
        f"{S3_PROCESSED_PREFIX}X_train.parquet",
        f"{S3_PROCESSED_PREFIX}X_test.parquet",
        f"{S3_PROCESSED_PREFIX}y_train.parquet",
        f"{S3_PROCESSED_PREFIX}y_test.parquet",
        f"{S3_PROCESSED_PREFIX}encoder.joblib",
        f"{S3_PROCESSED_PREFIX}feature_names.joblib",
    ]

    missing = [k for k in required_keys if not file_exists(k)]
    if not missing:
        logger.info("All preprocessed data found on MinIO.")
        return

    logger.info("Missing data on MinIO: %s. Running full download + preprocess.", missing)

    import io
    import tempfile
    from pathlib import Path

    import joblib
    import pandas as pd

    from src.data.download import download_dataset
    from src.data.preprocess import run_preprocessing_pipeline
    from src.data.s3_storage import upload_bytes, upload_dataframe

    # Download from Kaggle to temp dir
    tmpdir = Path(tempfile.mkdtemp(prefix="kaggle_"))
    csv_path = download_dataset(output_dir=tmpdir)
    df = pd.read_csv(csv_path)
    logger.info("Downloaded %d rows from Kaggle.", len(df))

    # Upload raw
    upload_dataframe(df, key="raw/kc_house_data.parquet")

    # Preprocess
    data = run_preprocessing_pipeline(df)

    # Upload processed splits
    upload_dataframe(data["X_train"], key=f"{S3_PROCESSED_PREFIX}X_train.parquet")
    upload_dataframe(data["X_test"], key=f"{S3_PROCESSED_PREFIX}X_test.parquet")

    y_train_df = pd.DataFrame({"target": data["y_train"]})
    y_test_df = pd.DataFrame({"target": data["y_test"]})
    upload_dataframe(y_train_df, key=f"{S3_PROCESSED_PREFIX}y_train.parquet")
    upload_dataframe(y_test_df, key=f"{S3_PROCESSED_PREFIX}y_test.parquet")

    buf = io.BytesIO()
    joblib.dump(data["encoder"], buf)
    upload_bytes(buf.getvalue(), key=f"{S3_PROCESSED_PREFIX}encoder.joblib")

    buf = io.BytesIO()
    joblib.dump(data["feature_names"], buf)
    upload_bytes(buf.getvalue(), key=f"{S3_PROCESSED_PREFIX}feature_names.joblib")

    logger.info("Preprocessed data uploaded to MinIO.")


def _train_model(**kwargs):
    """Load preprocessed data from MinIO and train the model."""
    import io

    import joblib

    from src.data.s3_storage import download_bytes, download_dataframe
    from src.model.train import train_model

    # Load splits from MinIO
    X_train = download_dataframe(key=f"{S3_PROCESSED_PREFIX}X_train.parquet")
    X_test = download_dataframe(key=f"{S3_PROCESSED_PREFIX}X_test.parquet")

    y_train = download_dataframe(key=f"{S3_PROCESSED_PREFIX}y_train.parquet")["target"]
    y_test = download_dataframe(key=f"{S3_PROCESSED_PREFIX}y_test.parquet")["target"]

    # Load encoder (needed for the model artifact context, not used directly here)
    joblib.load(io.BytesIO(download_bytes(key=f"{S3_PROCESSED_PREFIX}encoder.joblib")))
    feature_names = joblib.load(
        io.BytesIO(download_bytes(key=f"{S3_PROCESSED_PREFIX}feature_names.joblib"))
    )

    logger.info(
        "Loaded data from MinIO: %d train / %d test, %d features",
        len(X_train),
        len(X_test),
        len(feature_names),
    )

    model, metrics = train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize=True,
        n_trials=10,
    )

    logger.info("Training complete. Metrics: %s", metrics)
    return metrics


def _compare_and_promote(**kwargs):
    """Compare new model with previous best and promote only if better.

    Uses compare_models() from evaluate.py to check RMSE on price scale.
    If the new model is worse, its MLflow version is NOT promoted and a
    warning is logged. The previous model stays as the active one.
    """
    import mlflow

    from src.config import MLFLOW_TRACKING_URI
    from src.model.evaluate import compare_models

    ti = kwargs["ti"]
    new_metrics = ti.xcom_pull(task_ids="train_model")

    if not new_metrics:
        logger.warning("No metrics from training. Skipping comparison.")
        return {"promoted": False, "reason": "no_metrics"}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get all versions of the registered model
    model_name = "house-price-xgboost"
    try:
        versions = client.get_latest_versions(model_name)
    except Exception:
        logger.info("No registered model found. New model is the first — promoting.")
        return {"promoted": True, "metrics": new_metrics, "reason": "first_model"}

    if len(versions) < 2:
        logger.info("Only one model version exists. Promoting by default.")
        return {"promoted": True, "metrics": new_metrics, "reason": "first_model"}

    # The latest version is the one we just trained; the previous is the one before
    sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    latest_version = sorted_versions[0]
    previous_version = sorted_versions[1]

    # Fetch previous model metrics from its MLflow run
    try:
        prev_run = client.get_run(previous_version.run_id)
        previous_metrics = prev_run.data.metrics
    except Exception as e:
        logger.warning("Could not fetch previous model metrics: %s. Promoting new model.", e)
        return {"promoted": True, "metrics": new_metrics, "reason": "previous_unavailable"}

    # Compare
    is_better = compare_models(
        current_metrics=new_metrics,
        previous_metrics=previous_metrics,
        primary_metric="rmse_price",
        lower_is_better=True,
    )

    if is_better:
        logger.info(
            "New model (v%s) is BETTER than previous (v%s). Keeping as latest.",
            latest_version.version,
            previous_version.version,
        )
        return {"promoted": True, "metrics": new_metrics, "reason": "better"}
    else:
        # Delete the worse version from the registry to keep it clean
        logger.warning(
            "New model (v%s) is WORSE than previous (v%s). Deleting new version from registry.",
            latest_version.version,
            previous_version.version,
        )
        client.delete_model_version(name=model_name, version=latest_version.version)
        return {
            "promoted": False,
            "metrics": new_metrics,
            "reason": "worse",
            "previous_rmse": previous_metrics.get("rmse_price"),
            "new_rmse": new_metrics.get("rmse_price"),
        }


def _send_success_alert(**kwargs):
    """Send success email after training completes."""
    from src.utils.alerting import send_training_success_alert

    ti = kwargs["ti"]
    comparison = ti.xcom_pull(task_ids="compare_and_promote")
    metrics = ti.xcom_pull(task_ids="train_model")

    if not metrics:
        logger.warning("No metrics available; skipping success alert.")
        return

    if comparison and comparison.get("promoted"):
        send_training_success_alert(metrics)
        logger.info("Success alert sent — new model promoted.")
    else:
        reason = comparison.get("reason", "unknown") if comparison else "unknown"
        logger.info("Model NOT promoted (reason: %s). Alert sent with info.", reason)
        send_training_success_alert(metrics)


def _send_failure_alert(context):
    """Callback on task failure: send failure email."""
    from src.utils.alerting import send_training_failure_alert

    exception = context.get("exception")
    task_id = context.get("task_instance").task_id
    error_msg = f"Task '{task_id}' failed: {exception}"
    send_training_failure_alert(error_msg)


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="ML training pipeline: ensure data on MinIO, train, compare with previous, alert",
    schedule=None,  # Triggered manually or by data_pipeline DAG
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "mlops", "minio"],
) as dag:
    ensure_data = PythonOperator(
        task_id="ensure_data_on_minio",
        python_callable=_ensure_data_on_minio,
        on_failure_callback=_send_failure_alert,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        on_failure_callback=_send_failure_alert,
    )

    compare = PythonOperator(
        task_id="compare_and_promote",
        python_callable=_compare_and_promote,
        on_failure_callback=_send_failure_alert,
    )

    alert = PythonOperator(
        task_id="send_success_alert",
        python_callable=_send_success_alert,
    )

    ensure_data >> train >> compare >> alert
