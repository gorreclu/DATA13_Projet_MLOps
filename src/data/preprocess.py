"""Data preprocessing pipeline for House Sales prediction.

Pipeline steps:
    1. Cleaning: parse dates, remove outliers, handle missing values
    2. Feature engineering: house_age, is_renovated, ratios, neighborhood features
    3. Categorical encoding: target encoding for zipcode
    4. Multicollinearity: drop redundant features
    5. Target transformation: log1p(price)
"""

import logging

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COL, TEST_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: parse dates, drop id, handle outliers and missing values."""
    df = df.copy()

    # Parse date column -> extract year_sold, month_sold
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["year_sold"] = df["date"].dt.year
    df["month_sold"] = df["date"].dt.month

    # Drop id and date (no predictive value after feature extraction)
    df = df.drop(columns=["id", "date"], errors="ignore")

    # Remove the 33-bedroom outlier (known data entry error)
    n_before = len(df)
    df = df[df["bedrooms"] < 30]
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info("Removed %d outlier rows (bedrooms >= 30).", n_removed)

    # Handle missing waterfront values -> default 0 (no waterfront)
    if df["waterfront"].isna().any():
        n_missing = df["waterfront"].isna().sum()
        df["waterfront"] = df["waterfront"].fillna(0).astype(int)
        logger.info("Imputed %d missing waterfront values to 0.", n_missing)

    # Ensure correct dtypes
    int_cols = ["bedrooms", "floors", "waterfront", "view", "condition", "grade"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    logger.info("Cleaning done. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features that improve model performance."""
    df = df.copy()

    # House age at time of sale
    df["house_age"] = df["year_sold"] - df["yr_built"]

    # Renovation features
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_renovation"] = np.where(
        df["yr_renovated"] > 0,
        df["year_sold"] - df["yr_renovated"],
        df["house_age"],  # If never renovated, use house age
    )

    # Basement
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # Living space vs neighbors ratio
    df["living_vs_neighbors"] = df["sqft_living"] / df["sqft_living15"].replace(0, 1)

    # Lot vs neighbors ratio
    df["lot_vs_neighbors"] = df["sqft_lot"] / df["sqft_lot15"].replace(0, 1)

    # Space ratios
    df["sqft_living_lot_ratio"] = df["sqft_living"] / df["sqft_lot"].replace(0, 1)
    df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

    # Above ground ratio (what fraction of living space is above ground)
    df["above_ground_ratio"] = df["sqft_above"] / df["sqft_living"].replace(0, 1)

    # Drop columns that are now redundant or encoded into new features
    drop_after_engineering = [
        "sqft_above",  # redundant with sqft_living + has_basement
        "sqft_basement",  # replaced by has_basement
        "yr_built",  # replaced by house_age
        "yr_renovated",  # replaced by is_renovated + years_since_renovation
        "sqft_living15",  # replaced by living_vs_neighbors
        "sqft_lot15",  # replaced by lot_vs_neighbors
    ]
    df = df.drop(columns=drop_after_engineering, errors="ignore")

    logger.info("Feature engineering done. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Target Encoding for zipcode
# ---------------------------------------------------------------------------
def encode_zipcode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """Apply target encoding to zipcode using only training data.

    This avoids data leakage: the encoder learns statistics only from the
    train set and transforms both train and test.
    """
    encoder = TargetEncoder(cols=["zipcode"], smoothing=10)
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["zipcode"] = encoder.fit_transform(df_train[["zipcode"]], df_train[target_col])[
        "zipcode"
    ]
    df_test["zipcode"] = encoder.transform(df_test[["zipcode"]])["zipcode"]

    logger.info("Zipcode target encoding applied.")
    return df_train, df_test, encoder


# ---------------------------------------------------------------------------
# Step 4 — Target Transformation (log)
# ---------------------------------------------------------------------------
def transform_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply log1p to the target and separate features from target."""
    df = df.copy()
    y = np.log1p(df[target_col])
    X = df.drop(columns=[target_col])
    return X, y


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """Convert log-transformed predictions back to original price scale."""
    return np.expm1(y_log)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(
    df: pd.DataFrame,
) -> dict:
    """Run the full preprocessing pipeline.

    Returns:
        dict with keys:
            - X_train, X_test: feature DataFrames
            - y_train, y_test: log-transformed target Series
            - encoder: fitted TargetEncoder for zipcode
            - feature_names: list of feature column names
    """
    logger.info("Starting preprocessing pipeline. Input shape: %s", df.shape)

    # Step 1 — Cleaning
    df = clean_data(df)

    # Step 2 — Feature Engineering
    df = engineer_features(df)

    # Split BEFORE encoding to avoid data leakage
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    logger.info("Train/test split: %d / %d", len(df_train), len(df_test))

    # Step 3 — Target Encoding (fit on train only)
    df_train, df_test, encoder = encode_zipcode(df_train, df_test)

    # Step 4 — Target Transformation + Feature/Target separation
    X_train, y_train = transform_target(df_train)
    X_test, y_test = transform_target(df_test)

    feature_names = list(X_train.columns)
    logger.info("Pipeline complete. %d features: %s", len(feature_names), feature_names)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoder": encoder,
        "feature_names": feature_names,
    }


def preprocess_single_input(
    data: dict,
    encoder: TargetEncoder,
    feature_names: list[str],
) -> pd.DataFrame:
    """Preprocess a single input for inference (API / WebApp).

    Args:
        data: dict with raw feature values (as received from the user).
        encoder: fitted TargetEncoder for zipcode.
        feature_names: expected feature column order.

    Returns:
        DataFrame with one row, ready for model.predict().
    """
    df = pd.DataFrame([data])

    # Simulate the same feature engineering
    # The user provides: bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    # waterfront, view, condition, grade, zipcode, lat, long,
    # year_sold, month_sold, yr_built, yr_renovated, sqft_above, sqft_basement,
    # sqft_living15, sqft_lot15
    df["house_age"] = df["year_sold"] - df["yr_built"]
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["years_since_renovation"] = np.where(
        df["yr_renovated"] > 0,
        df["year_sold"] - df["yr_renovated"],
        df["house_age"],
    )
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    df["living_vs_neighbors"] = df["sqft_living"] / df["sqft_living15"].replace(0, 1)
    df["lot_vs_neighbors"] = df["sqft_lot"] / df["sqft_lot15"].replace(0, 1)
    df["sqft_living_lot_ratio"] = df["sqft_living"] / df["sqft_lot"].replace(0, 1)
    df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["above_ground_ratio"] = df["sqft_above"] / df["sqft_living"].replace(0, 1)

    # Drop raw columns that were removed during training
    drop_cols = [
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "sqft_living15",
        "sqft_lot15",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Target-encode zipcode
    df["zipcode"] = encoder.transform(df[["zipcode"]])["zipcode"]

    # Ensure correct column order and fill missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df
