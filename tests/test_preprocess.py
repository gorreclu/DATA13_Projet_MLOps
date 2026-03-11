"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    clean_data,
    encode_zipcode,
    engineer_features,
    inverse_transform_target,
    preprocess_single_input,
    run_preprocessing_pipeline,
    transform_target,
)


@pytest.fixture
def sample_raw_df():
    """Create a minimal DataFrame mimicking the raw King County dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "date": [
                "20141013T000000",
                "20141209T000000",
                "20150225T000000",
                "20141209T000000",
                "20150218T000000",
            ],
            "price": [221900, 538000, 180000, 604000, 510000],
            "bedrooms": [3, 3, 2, 4, 3],
            "bathrooms": [1.0, 2.25, 1.0, 3.0, 2.0],
            "sqft_living": [1180, 2570, 770, 1960, 1680],
            "sqft_lot": [5650, 7242, 10000, 5000, 8080],
            "floors": [1.0, 2.0, 1.0, 1.0, 1.0],
            "waterfront": [0, 0, 0, 0, 0],
            "view": [0, 0, 0, 0, 0],
            "condition": [3, 3, 3, 5, 3],
            "grade": [7, 7, 6, 7, 8],
            "sqft_above": [1180, 2170, 770, 1050, 1680],
            "sqft_basement": [0, 400, 0, 910, 0],
            "yr_built": [1955, 1951, 1933, 1965, 1987],
            "yr_renovated": [0, 1991, 0, 0, 0],
            "zipcode": [98178, 98125, 98028, 98136, 98074],
            "lat": [47.5112, 47.7210, 47.7379, 47.5208, 47.6168],
            "long": [-122.257, -122.319, -122.233, -122.393, -122.045],
            "sqft_living15": [1340, 1690, 2720, 1360, 1800],
            "sqft_lot15": [5650, 7639, 8062, 5000, 7503],
        }
    )


class TestCleanData:
    """Tests for the clean_data function."""

    def test_drops_id_and_date(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert "id" not in result.columns
        assert "date" not in result.columns

    def test_extracts_year_month(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert "year_sold" in result.columns
        assert "month_sold" in result.columns
        assert result["year_sold"].iloc[0] == 2014
        assert result["month_sold"].iloc[0] == 10

    def test_removes_bedroom_outlier(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "bedrooms"] = 33
        result = clean_data(df)
        assert len(result) == len(sample_raw_df) - 1
        assert (result["bedrooms"] < 30).all()

    def test_imputes_waterfront_nan(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "waterfront"] = np.nan
        result = clean_data(df)
        assert result["waterfront"].isna().sum() == 0
        assert result["waterfront"].iloc[0] == 0

    def test_output_shape(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        # Should have original columns minus id, date + year_sold, month_sold
        expected_cols = len(sample_raw_df.columns) - 2 + 2  # -id -date +year_sold +month_sold
        assert result.shape[1] == expected_cols
        assert result.shape[0] == len(sample_raw_df)


class TestEngineerFeatures:
    """Tests for the engineer_features function."""

    def test_creates_expected_features(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        result = engineer_features(cleaned)

        expected_new = [
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
        for feat in expected_new:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_drops_redundant_columns(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        result = engineer_features(cleaned)

        dropped = [
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "sqft_living15",
            "sqft_lot15",
        ]
        for col in dropped:
            assert col not in result.columns, f"Should have dropped: {col}"

    def test_house_age_calculation(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        result = engineer_features(cleaned)
        # First row: year_sold=2014, yr_built=1955 -> age=59
        assert result["house_age"].iloc[0] == 2014 - 1955

    def test_is_renovated(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        result = engineer_features(cleaned)
        # Second row has yr_renovated=1991 -> is_renovated=1
        assert result["is_renovated"].iloc[1] == 1
        # First row has yr_renovated=0 -> is_renovated=0
        assert result["is_renovated"].iloc[0] == 0

    def test_has_basement(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        result = engineer_features(cleaned)
        # Second row: sqft_basement=400 -> has_basement=1
        assert result["has_basement"].iloc[1] == 1
        # First row: sqft_basement=0 -> has_basement=0
        assert result["has_basement"].iloc[0] == 0


class TestTargetEncoding:
    """Tests for zipcode target encoding."""

    def test_encode_zipcode_no_leakage(self, sample_raw_df):
        """Encoder should be fit only on train data."""
        cleaned = clean_data(sample_raw_df)
        engineered = engineer_features(cleaned)

        df_train = engineered.iloc[:3].copy()
        df_test = engineered.iloc[3:].copy()

        df_train_enc, df_test_enc, encoder = encode_zipcode(df_train, df_test)

        # Zipcode should now be float (encoded), not integer
        assert df_train_enc["zipcode"].dtype == np.float64
        assert df_test_enc["zipcode"].dtype == np.float64

    def test_encoder_is_fitted(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        engineered = engineer_features(cleaned)

        df_train = engineered.iloc[:3].copy()
        df_test = engineered.iloc[3:].copy()

        _, _, encoder = encode_zipcode(df_train, df_test)
        # Should be able to transform new data
        assert hasattr(encoder, "mapping")


class TestTargetTransformation:
    """Tests for log target transformation."""

    def test_transform_target(self, sample_raw_df):
        cleaned = clean_data(sample_raw_df)
        X, y = transform_target(cleaned)
        assert "price" not in X.columns
        assert len(y) == len(X)
        # log1p(price) should be positive
        assert (y > 0).all()

    def test_inverse_transform_roundtrip(self):
        prices = np.array([221900, 538000, 180000])
        log_prices = np.log1p(prices)
        recovered = inverse_transform_target(log_prices)
        np.testing.assert_array_almost_equal(prices, recovered, decimal=0)


class TestFullPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_pipeline_output_keys(self, sample_raw_df):
        result = run_preprocessing_pipeline(sample_raw_df)
        expected_keys = {"X_train", "X_test", "y_train", "y_test", "encoder", "feature_names"}
        assert set(result.keys()) == expected_keys

    def test_pipeline_no_target_leakage(self, sample_raw_df):
        result = run_preprocessing_pipeline(sample_raw_df)
        assert "price" not in result["X_train"].columns
        assert "price" not in result["X_test"].columns

    def test_pipeline_feature_names_match(self, sample_raw_df):
        result = run_preprocessing_pipeline(sample_raw_df)
        assert list(result["X_train"].columns) == result["feature_names"]
        assert list(result["X_test"].columns) == result["feature_names"]

    def test_pipeline_train_test_sizes(self, sample_raw_df):
        result = run_preprocessing_pipeline(sample_raw_df)
        total = len(result["X_train"]) + len(result["X_test"])
        # Original has 5 rows
        assert total == 5


class TestPreprocessSingleInput:
    """Tests for single-input preprocessing (API inference path)."""

    def test_single_input_produces_correct_columns(self, sample_raw_df):
        pipeline_result = run_preprocessing_pipeline(sample_raw_df)
        encoder = pipeline_result["encoder"]
        feature_names = pipeline_result["feature_names"]

        input_data = {
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
            "zipcode": 98178,
            "lat": 47.5112,
            "long": -122.257,
            "sqft_living15": 1800,
            "sqft_lot15": 5000,
            "year_sold": 2015,
            "month_sold": 6,
        }

        result = preprocess_single_input(input_data, encoder, feature_names)
        assert list(result.columns) == feature_names
        assert len(result) == 1
