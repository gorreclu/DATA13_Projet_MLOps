"""Tests for model training, evaluation, and prediction."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.model.evaluate import compare_models, evaluate_model
from src.model.predict import predict, predict_single


@pytest.fixture
def dummy_train_data():
    """Create small dummy train/test datasets."""
    np.random.seed(42)
    n_train, n_test = 100, 20
    n_features = 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y_train = pd.Series(np.random.uniform(11, 14, n_train), name="price")  # log scale

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y_test = pd.Series(np.random.uniform(11, 14, n_test), name="price")

    return X_train, y_train, X_test, y_test


@pytest.fixture
def trained_model(dummy_train_data):
    """Train a quick XGBoost model for testing."""
    X_train, y_train, X_test, y_test = dummy_train_data
    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_expected_metrics(self, trained_model, dummy_train_data):
        _, _, X_test, y_test = dummy_train_data
        metrics = evaluate_model(trained_model, X_test, y_test)

        expected_keys = {
            "rmse_log",
            "mae_log",
            "r2_log",
            "rmse_price",
            "mae_price",
            "r2_price",
            "mape",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_numeric(self, trained_model, dummy_train_data):
        _, _, X_test, y_test = dummy_train_data
        metrics = evaluate_model(trained_model, X_test, y_test)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric: {type(value)}"

    def test_rmse_positive(self, trained_model, dummy_train_data):
        _, _, X_test, y_test = dummy_train_data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert metrics["rmse_log"] > 0
        assert metrics["rmse_price"] > 0

    def test_mape_positive(self, trained_model, dummy_train_data):
        _, _, X_test, y_test = dummy_train_data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert metrics["mape"] >= 0


class TestCompareModels:
    """Tests for the compare_models function."""

    def test_first_model_always_wins(self):
        current = {"rmse_price": 50000.0}
        assert compare_models(current, None) is True

    def test_lower_is_better(self):
        current = {"rmse_price": 40000.0}
        previous = {"rmse_price": 50000.0}
        assert compare_models(current, previous) is True

    def test_higher_is_worse(self):
        current = {"rmse_price": 60000.0}
        previous = {"rmse_price": 50000.0}
        assert compare_models(current, previous) is False

    def test_higher_is_better_r2(self):
        current = {"r2_price": 0.92}
        previous = {"r2_price": 0.88}
        assert (
            compare_models(current, previous, primary_metric="r2_price", lower_is_better=False)
            is True
        )


class TestPredict:
    """Tests for prediction functions."""

    def test_predict_returns_real_prices(self, trained_model, dummy_train_data):
        _, _, X_test, _ = dummy_train_data
        prices = predict(trained_model, X_test)
        assert len(prices) == len(X_test)
        # Prices should be positive (expm1 of log-scale predictions)
        assert (prices > 0).all()

    def test_predict_single_returns_float(self, trained_model, dummy_train_data):
        _, _, X_test, _ = dummy_train_data
        price = predict_single(trained_model, X_test.iloc[:1])
        assert isinstance(price, float)
        assert price > 0

    def test_predict_output_scale(self, trained_model, dummy_train_data):
        """Predictions should be on real price scale (not log)."""
        _, _, X_test, _ = dummy_train_data
        prices = predict(trained_model, X_test)
        # If y is ~12 on log scale, real prices are ~exp(12) ~ 162,000
        # Allow a wide range since dummy data is random
        assert prices.max() > 100  # At minimum, should be larger than log-scale values
