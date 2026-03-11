"""MinIO S3 storage utilities for the House Price MLOps project.

Provides helper functions to upload/download DataFrames, files, and bytes
to/from a MinIO S3-compatible object store.
"""

import io
import logging

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

from src.config import (
    MINIO_BUCKET_DATA,
    MINIO_ENDPOINT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
)

logger = logging.getLogger(__name__)


def get_s3_client():
    """Create and return a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(bucket: str = MINIO_BUCKET_DATA) -> None:
    """Create the bucket if it does not already exist."""
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
        logger.debug("Bucket '%s' already exists.", bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        logger.info("Created bucket '%s'.", bucket)


def upload_dataframe(
    df: pd.DataFrame,
    key: str,
    bucket: str = MINIO_BUCKET_DATA,
    fmt: str = "parquet",
) -> str:
    """Upload a pandas DataFrame to MinIO as parquet or CSV.

    Args:
        df: DataFrame to upload.
        key: S3 object key (path inside the bucket), e.g. 'raw/data.parquet'.
        bucket: Target bucket name.
        fmt: Format — 'parquet' (default) or 'csv'.

    Returns:
        The S3 URI of the uploaded object.
    """
    ensure_bucket(bucket)
    client = get_s3_client()

    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
    elif fmt == "csv":
        df.to_csv(buf, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    buf.seek(0)
    client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded DataFrame (%d rows) to %s", len(df), uri)
    return uri


def download_dataframe(
    key: str,
    bucket: str = MINIO_BUCKET_DATA,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Download a DataFrame from MinIO.

    Args:
        key: S3 object key.
        bucket: Source bucket name.
        fmt: Format — 'parquet' (default) or 'csv'.

    Returns:
        The downloaded DataFrame.
    """
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()

    if fmt == "parquet":
        df = pd.read_parquet(io.BytesIO(data))
    elif fmt == "csv":
        df = pd.read_csv(io.BytesIO(data))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info("Downloaded DataFrame (%d rows) from s3://%s/%s", len(df), bucket, key)
    return df


def upload_file(local_path: str, key: str, bucket: str = MINIO_BUCKET_DATA) -> str:
    """Upload a local file to MinIO.

    Args:
        local_path: Path to the local file.
        key: S3 object key.
        bucket: Target bucket name.

    Returns:
        The S3 URI of the uploaded object.
    """
    ensure_bucket(bucket)
    client = get_s3_client()
    client.upload_file(local_path, bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded file %s to %s", local_path, uri)
    return uri


def download_file(key: str, local_path: str, bucket: str = MINIO_BUCKET_DATA) -> str:
    """Download a file from MinIO to a local path.

    Args:
        key: S3 object key.
        local_path: Destination local path.
        bucket: Source bucket name.

    Returns:
        The local path.
    """
    client = get_s3_client()
    client.download_file(bucket, key, local_path)
    logger.info("Downloaded s3://%s/%s to %s", bucket, key, local_path)
    return local_path


def upload_bytes(data: bytes, key: str, bucket: str = MINIO_BUCKET_DATA) -> str:
    """Upload raw bytes to MinIO.

    Args:
        data: Bytes content.
        key: S3 object key.
        bucket: Target bucket name.

    Returns:
        The S3 URI of the uploaded object.
    """
    ensure_bucket(bucket)
    client = get_s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded %d bytes to %s", len(data), uri)
    return uri


def download_bytes(key: str, bucket: str = MINIO_BUCKET_DATA) -> bytes:
    """Download raw bytes from MinIO.

    Args:
        key: S3 object key.
        bucket: Source bucket name.

    Returns:
        The downloaded bytes.
    """
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()
    logger.info("Downloaded %d bytes from s3://%s/%s", len(data), bucket, key)
    return data


def file_exists(key: str, bucket: str = MINIO_BUCKET_DATA) -> bool:
    """Check if an object exists in MinIO.

    Args:
        key: S3 object key.
        bucket: Bucket name.

    Returns:
        True if the object exists, False otherwise.
    """
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
