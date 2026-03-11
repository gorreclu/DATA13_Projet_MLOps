"""CLI entry points for model training."""

import logging

import pandas as pd
from dotenv import load_dotenv

from src.config import RAW_CSV_FILENAME, RAW_DATA_DIR
from src.data.preprocess import run_preprocessing_pipeline
from src.model.train import train_pipeline


def train():
    """Full training with Optuna hyperparameter optimization."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    df = pd.read_csv(RAW_DATA_DIR / RAW_CSV_FILENAME)
    data = run_preprocessing_pipeline(df)
    model, metrics = train_pipeline(data, optimize=True)
    print("Training complete!")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def train_quick():
    """Quick training without Optuna optimization."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    df = pd.read_csv(RAW_DATA_DIR / RAW_CSV_FILENAME)
    data = run_preprocessing_pipeline(df)
    model, metrics = train_pipeline(data, optimize=False)
    print("Training complete!")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
