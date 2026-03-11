"""FastAPI serving API for house price prediction."""

import logging
import time
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from src.api.schemas import HealthResponse, HouseFeatures, PredictionResponse
from src.config import MLFLOW_TRACKING_URI
from src.data.preprocess import preprocess_single_input
from src.model.predict import get_model, predict_single

logger = logging.getLogger(__name__)

# --- Prometheus metrics ---
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)
MODEL_RMSE = Gauge(
    "model_rmse_price",
    "RMSE of the currently loaded model",
)
MODEL_R2 = Gauge(
    "model_r2_price",
    "R2 score of the currently loaded model",
)
DATA_ROWS = Gauge(
    "data_rows_processed",
    "Number of data rows used for training",
)
TRAINING_RUNS = Gauge(
    "training_runs_total",
    "Total number of training runs recorded in MLflow",
)
MODEL_MAE = Gauge(
    "model_mae_price",
    "MAE of the currently loaded model",
)
MODEL_MAPE = Gauge(
    "model_mape",
    "MAPE of the currently loaded model (percentage)",
)

# Global state for encoder and feature names
_encoder = None
_feature_names = None


def _load_artifacts():
    """Load the zipcode encoder and feature names from MinIO S3."""
    global _encoder, _feature_names
    if _encoder is not None and _feature_names is not None:
        return

    try:
        import io

        import joblib

        from src.data.s3_storage import download_bytes

        encoder_bytes = download_bytes("processed/encoder.joblib")
        features_bytes = download_bytes("processed/feature_names.joblib")

        _encoder = joblib.load(io.BytesIO(encoder_bytes))
        _feature_names = joblib.load(io.BytesIO(features_bytes))
        logger.info("Artifacts loaded from MinIO S3: encoder + %d features", len(_feature_names))

    except Exception as e:
        logger.error("Failed to load artifacts from MinIO: %s", e)
        raise


def _load_model_metrics(client, run_id):
    """Fetch model metrics from the MLflow run and expose them to Prometheus."""
    try:
        run = client.get_run(run_id)
        run_metrics = run.data.metrics

        rmse = (
            run_metrics.get("rmse_price") or run_metrics.get("rmse") or run_metrics.get("test_rmse")
        )
        r2 = run_metrics.get("r2_price") or run_metrics.get("r2") or run_metrics.get("test_r2")
        mae = run_metrics.get("mae_price") or run_metrics.get("mae")
        mape = run_metrics.get("mape")

        if rmse is not None:
            MODEL_RMSE.set(float(rmse))
            logger.info("Prometheus gauge model_rmse_price set to %.2f", rmse)
        if r2 is not None:
            MODEL_R2.set(float(r2))
            logger.info("Prometheus gauge model_r2_price set to %.4f", r2)
        if mae is not None:
            MODEL_MAE.set(float(mae))
            logger.info("Prometheus gauge model_mae_price set to %.2f", mae)
        if mape is not None:
            MODEL_MAPE.set(float(mape))
            logger.info("Prometheus gauge model_mape set to %.2f", mape)

        # Count total training runs for the experiment
        experiment_name = run.info.experiment_id
        runs = client.search_runs(
            experiment_ids=[experiment_name],
            filter_string="attributes.status = 'FINISHED'",
        )
        TRAINING_RUNS.set(len(runs))
        logger.info("Prometheus gauge training_runs_total set to %d", len(runs))

        # Data rows from logged params
        params = run.data.params
        n_rows = (
            params.get("n_train_samples")
            or params.get("n_train_rows")
            or params.get("train_size")
            or params.get("n_samples")
        )
        if n_rows is not None:
            DATA_ROWS.set(float(n_rows))
            logger.info("Prometheus gauge data_rows_processed set to %s", n_rows)

    except Exception as e:
        logger.warning("Could not load model metrics from MLflow: %s", e)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model and artifacts on startup."""
    logger.info("Loading model and artifacts...")
    try:
        get_model()
        _load_artifacts()
        logger.info("API ready.")
    except Exception as e:
        logger.error("Startup failed: %s", e)

    # Load Prometheus metrics from MLflow (non-blocking)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("house-price-xgboost")
        if versions:
            _load_model_metrics(client, versions[-1].run_id)
    except Exception as e:
        logger.warning("Could not load MLflow metrics: %s", e)

    yield


# --- App ---
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predict house prices in King County, WA using XGBoost. Part of the DATA713 MLOps project."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    model_loaded = _encoder is not None and _feature_names is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        mlflow_uri=MLFLOW_TRACKING_URI,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(features: HouseFeatures):
    """Predict the price of a house given its features."""
    PREDICTION_COUNT.inc()
    start_time = time.time()

    try:
        _load_artifacts()
        model = get_model()

        # Preprocess input
        X = preprocess_single_input(
            data=features.model_dump(),
            encoder=_encoder,
            feature_names=_feature_names,
        )

        # Predict
        price = predict_single(model, X)

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        formatted = f"${price:,.0f}"
        logger.info("Prediction: %s (latency: %.3fs)", formatted, latency)

        return PredictionResponse(
            predicted_price=round(price, 2),
            formatted_price=formatted,
        )

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
