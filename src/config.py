"""Centralized configuration for the House Price MLOps project."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"

# --- Kaggle ---
KAGGLE_DATASET = "harlfoxem/housesalesprediction"
RAW_CSV_FILENAME = "kc_house_data.csv"

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://89.58.44.97:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price-prediction")

# --- PostgreSQL ---
POSTGRES_USER = os.getenv("POSTGRES_USER", os.getenv("PG_USER", "mlops"))
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", os.getenv("PG_PASSWORD", "mlops_password"))
POSTGRES_DB = os.getenv("POSTGRES_DB", os.getenv("PG_DB", "mlops"))
POSTGRES_HOST = os.getenv("PG_HOST", os.getenv("POSTGRES_HOST", "localhost"))

_raw_pg_port = os.getenv("PG_PORT", os.getenv("POSTGRES_PORT", "5432"))
# K8s injects POSTGRES_PORT as 'tcp://...' for service named 'postgres'
if _raw_pg_port.startswith("tcp://"):
    _raw_pg_port = "5432"
POSTGRES_PORT = int(_raw_pg_port)

# --- SMTP Alerting ---
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
ALERT_RECIPIENTS = [r.strip() for r in os.getenv("ALERT_RECIPIENTS", "").split(",") if r.strip()]

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
_raw_api_port = os.getenv("APP_API_PORT", os.getenv("API_PORT", "8000"))
if isinstance(_raw_api_port, str) and _raw_api_port.startswith("tcp://"):
    _raw_api_port = "8000"
API_PORT = int(_raw_api_port)

# --- Webapp ---
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
_raw_webapp_port = os.getenv("APP_WEBAPP_PORT", os.getenv("WEBAPP_PORT", "7860"))
if isinstance(_raw_webapp_port, str) and _raw_webapp_port.startswith("tcp://"):
    _raw_webapp_port = "7860"
WEBAPP_PORT = int(_raw_webapp_port)

# --- Model parameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OPTUNA_N_TRIALS = 10

# --- MinIO S3 ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_BUCKET_DATA = os.getenv("MINIO_BUCKET_DATA", "mlops-data")

# --- Feature lists ---
TARGET_COL = "price"
DROP_COLS = ["id", "date"]
