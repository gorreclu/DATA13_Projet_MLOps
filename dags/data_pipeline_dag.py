"""Airflow DAG: Scheduled data pipeline with MinIO S3 storage.

Runs weekly to check for new data, download from Kaggle, validate,
preprocess and store results on MinIO S3, then trigger training.
"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

logger = logging.getLogger(__name__)

# S3 keys for data stored in MinIO
S3_RAW_KEY = "raw/kc_house_data.parquet"
S3_PROCESSED_PREFIX = "processed/"

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _check_data_freshness(**kwargs):
    """Check if raw data exists on MinIO.

    Returns:
        'download_data' if data is missing or stale, 'trigger_training' otherwise.
    """
    from src.data.s3_storage import file_exists

    if not file_exists(S3_RAW_KEY):
        logger.info("Raw data not found on MinIO. Will download.")
        return "download_data"

    # Check object metadata for freshness
    import time

    from src.config import MINIO_BUCKET_DATA
    from src.data.s3_storage import get_s3_client

    client = get_s3_client()
    response = client.head_object(Bucket=MINIO_BUCKET_DATA, Key=S3_RAW_KEY)
    last_modified = response["LastModified"].timestamp()
    age_days = (time.time() - last_modified) / 86400

    if age_days > 7:
        logger.info("Raw data is %.1f days old. Triggering re-download.", age_days)
        return "download_data"

    logger.info("Raw data is fresh (%.1f days old). Triggering training only.", age_days)
    return "trigger_training"


def _download_data(**kwargs):
    """Download dataset from Kaggle and upload raw CSV to MinIO as parquet."""
    import os
    import tempfile
    from pathlib import Path

    import pandas as pd

    from src.data.download import download_dataset
    from src.data.s3_storage import upload_dataframe

    # Download from Kaggle to a temp directory (avoids permission issues)
    tmpdir = Path(tempfile.mkdtemp(prefix="kaggle_"))
    csv_path = download_dataset(output_dir=tmpdir)
    logger.info("Dataset downloaded to %s", csv_path)

    # Read and upload to MinIO
    df = pd.read_csv(csv_path)
    uri = upload_dataframe(df, key=S3_RAW_KEY)
    logger.info("Raw data uploaded to MinIO: %s (%d rows)", uri, len(df))

    # Clean up local file to save pod space
    if csv_path.exists():
        os.remove(csv_path)
        logger.info("Removed local file %s", csv_path)

    return uri


def _validate_data(**kwargs):
    """Download raw data from MinIO and run validation checks."""
    from src.data.s3_storage import download_dataframe

    df = download_dataframe(key=S3_RAW_KEY)

    checks = {
        "row_count": len(df) > 20000,
        "column_count": len(df.columns) == 21,
        "no_null_price": df["price"].isna().sum() == 0,
        "price_positive": (df["price"] > 0).all(),
        "has_required_cols": all(
            c in df.columns for c in ["price", "bedrooms", "bathrooms", "sqft_living", "zipcode"]
        ),
    }

    logger.info("Data validation results: %s", checks)

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise ValueError(f"Data validation failed on: {failed}")

    logger.info("All data validation checks passed (%d rows).", len(df))


def _preprocess_and_store(**kwargs):
    """Download raw data from MinIO, preprocess, and store results back to MinIO."""
    import io

    import joblib

    from src.data.preprocess import run_preprocessing_pipeline
    from src.data.s3_storage import download_dataframe, upload_bytes, upload_dataframe

    # Download raw data from MinIO
    df = download_dataframe(key=S3_RAW_KEY)
    logger.info("Loaded raw data from MinIO: %d rows", len(df))

    # Run preprocessing pipeline
    data = run_preprocessing_pipeline(df)

    # Upload processed splits to MinIO as parquet
    upload_dataframe(data["X_train"], key=f"{S3_PROCESSED_PREFIX}X_train.parquet")
    upload_dataframe(data["X_test"], key=f"{S3_PROCESSED_PREFIX}X_test.parquet")

    # Upload Series as parquet (convert to DataFrame first)
    import pandas as pd

    y_train_df = pd.DataFrame({"target": data["y_train"]})
    y_test_df = pd.DataFrame({"target": data["y_test"]})
    upload_dataframe(y_train_df, key=f"{S3_PROCESSED_PREFIX}y_train.parquet")
    upload_dataframe(y_test_df, key=f"{S3_PROCESSED_PREFIX}y_test.parquet")

    # Upload encoder and feature_names as joblib bytes
    buf = io.BytesIO()
    joblib.dump(data["encoder"], buf)
    upload_bytes(buf.getvalue(), key=f"{S3_PROCESSED_PREFIX}encoder.joblib")

    buf = io.BytesIO()
    joblib.dump(data["feature_names"], buf)
    upload_bytes(buf.getvalue(), key=f"{S3_PROCESSED_PREFIX}feature_names.joblib")

    logger.info(
        "Preprocessed data stored on MinIO: %d train / %d test samples, %d features",
        len(data["X_train"]),
        len(data["X_test"]),
        len(data["feature_names"]),
    )


with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    description=(
        "Scheduled data pipeline: check -> download -> validate"
        " -> preprocess -> store on MinIO -> trigger training"
    ),
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data", "pipeline", "mlops", "minio"],
) as dag:
    check_freshness = BranchPythonOperator(
        task_id="check_data_freshness",
        python_callable=_check_data_freshness,
    )

    download = PythonOperator(
        task_id="download_data",
        python_callable=_download_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        trigger_rule="none_failed_min_one_success",
    )

    preprocess_store = PythonOperator(
        task_id="preprocess_and_store",
        python_callable=_preprocess_and_store,
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        trigger_rule="none_failed_min_one_success",
    )

    # Branching: either download+validate+preprocess or go straight to training
    check_freshness >> [download, trigger_training]
    download >> validate >> preprocess_store >> trigger_training
