"""Download the House Sales dataset from Kaggle."""

import logging
import os
import zipfile
from pathlib import Path

from src.config import KAGGLE_DATASET, RAW_CSV_FILENAME, RAW_DATA_DIR

logger = logging.getLogger(__name__)


def download_dataset(output_dir: Path | None = None) -> Path:
    """Download and extract the House Sales dataset from Kaggle.

    Args:
        output_dir: Directory to save the dataset. Defaults to RAW_DATA_DIR.

    Returns:
        Path to the extracted CSV file.
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / RAW_CSV_FILENAME

    if csv_path.exists():
        logger.info("Dataset already exists at %s, skipping download.", csv_path)
        return csv_path

    # Ensure Kaggle credentials are available
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if kaggle_username and kaggle_key:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        logger.info("Downloading dataset '%s' from Kaggle...", KAGGLE_DATASET)
        api.dataset_download_files(KAGGLE_DATASET, path=str(output_dir), unzip=False)

        # Extract the zip
        zip_path = output_dir / "housesalesprediction.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(output_dir)
            zip_path.unlink()
            logger.info("Dataset extracted to %s", output_dir)

        if not csv_path.exists():
            # Sometimes the CSV has a different name inside the zip
            csv_files = list(output_dir.glob("*.csv"))
            if csv_files:
                csv_files[0].rename(csv_path)

        logger.info("Dataset ready at %s", csv_path)
        return csv_path

    except Exception as e:
        logger.error("Failed to download dataset: %s", e)
        raise


def main():
    """CLI entry point for dataset download."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from dotenv import load_dotenv

    load_dotenv()
    path = download_dataset()
    print(f"Dataset at: {path}")


if __name__ == "__main__":
    main()
