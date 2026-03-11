"""Tests for the FastAPI prediction API."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model():
    """Create a mock XGBoost model."""
    model = MagicMock()
    # Return log-scale predictions (like log1p of ~$500k)
    model.predict.return_value = np.array([13.12])
    return model


@pytest.fixture
def mock_encoder():
    """Create a mock TargetEncoder."""
    encoder = MagicMock()
    encoder.transform.return_value = pd.DataFrame({"zipcode": [12.5]})
    return encoder


@pytest.fixture
def mock_feature_names():
    """Return a realistic feature name list."""
    return [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "zipcode",
        "lat",
        "long",
        "year_sold",
        "month_sold",
        "house_age",
        "is_renovated",
        "years_since_renovation",
        "has_basement",
        "living_vs_neighbors",
        "lot_vs_neighbors",
        "sqft_living_lot_ratio",
        "bath_per_bed",
        "total_rooms",
        "above_ground_ratio",
    ]


@pytest.fixture
def client(mock_model, mock_encoder, mock_feature_names):
    """Create a test client with mocked dependencies."""
    with (
        patch("src.api.app.get_model", return_value=mock_model),
        patch("src.api.app._encoder", mock_encoder),
        patch("src.api.app._feature_names", mock_feature_names),
        patch("src.api.app._load_artifacts"),
    ):
        # Set the globals directly for health check
        import src.api.app as api_module
        from src.api.app import app

        api_module._encoder = mock_encoder
        api_module._feature_names = mock_feature_names

        yield TestClient(app)


@pytest.fixture
def sample_input():
    """Valid input payload for the /predict endpoint."""
    return {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "floors": 2.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 8,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "yr_built": 1990,
        "yr_renovated": 0,
        "zipcode": 98103,
        "lat": 47.6516,
        "long": -122.3480,
        "sqft_living15": 1800,
        "sqft_lot15": 5000,
        "year_sold": 2015,
        "month_sold": 6,
    }


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "mlflow_uri" in data

    def test_health_shows_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_200(self, client, sample_input):
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 200

    def test_predict_response_fields(self, client, sample_input):
        response = client.post("/predict", json=sample_input)
        data = response.json()
        assert "predicted_price" in data
        assert "formatted_price" in data

    def test_predict_price_is_positive(self, client, sample_input):
        response = client.post("/predict", json=sample_input)
        data = response.json()
        assert data["predicted_price"] > 0

    def test_predict_formatted_price_has_dollar(self, client, sample_input):
        response = client.post("/predict", json=sample_input)
        data = response.json()
        assert data["formatted_price"].startswith("$")

    def test_predict_invalid_bedrooms(self, client, sample_input):
        """Negative bedrooms should fail validation."""
        sample_input["bedrooms"] = -1
        response = client.post("/predict", json=sample_input)
        assert response.status_code == 422

    def test_predict_missing_required_field(self, client):
        """Missing required fields should fail."""
        response = client.post("/predict", json={"bedrooms": 3})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for the /metrics Prometheus endpoint."""

    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]


class TestDocsEndpoint:
    """Tests for API documentation endpoints."""

    def test_swagger_docs(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200
