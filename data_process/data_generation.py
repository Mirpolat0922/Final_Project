"""
Data Download and Loading Module
Downloads datasets from EPAM URLs and saves to data/raw directory.
"""

import os
import sys
import json
import logging
import pandas as pd
import zipfile
import requests
from pathlib import Path

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.getenv('CONF_PATH', 'settings.json')

from utils import get_project_dir, configure_logging

# Load configuration
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = get_project_dir(conf['general']['data_dir'])


def download_file(url, destination):
    """Download file from URL to destination."""
    logging.info(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Downloaded to {destination}")
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        raise


def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory."""
    logging.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"Extracted to {extract_to}")


def download_train_data():
    """Download and extract training data."""
    raw_dir = Path(DATA_DIR) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = raw_dir / 'final_project_train_dataset' / 'train.csv'

    if not train_csv_path.exists():
        train_zip_path = raw_dir / 'train.zip'
        download_file(conf['data_urls']['train_url'], train_zip_path)
        extract_zip(train_zip_path, raw_dir)
        train_zip_path.unlink()
        logging.info(f"Removed {train_zip_path}")
    else:
        logging.info(f"Training data already exists at {train_csv_path}")

    return pd.read_csv(train_csv_path)


def download_inference_data():
    """Download and extract inference data."""
    raw_dir = Path(DATA_DIR) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    inference_csv_path = raw_dir / 'final_project_inference_dataset' / 'inference.csv'

    if not inference_csv_path.exists():
        inference_zip_path = raw_dir / 'inference.zip'
        download_file(conf['data_urls']['inference_url'], inference_zip_path)
        extract_zip(inference_zip_path, raw_dir)
        inference_zip_path.unlink()
        logging.info(f"Removed {inference_zip_path}")
    else:
        logging.info(f"Inference data already exists at {inference_csv_path}")

    return pd.read_csv(inference_csv_path)


def main():
    """Download both datasets."""
    configure_logging()

    logging.info("=" * 60)
    logging.info("Downloading Training Data")
    logging.info("=" * 60)
    train_df = download_train_data()
    logging.info(f"Training data shape: {train_df.shape}")

    logging.info("\n" + "=" * 60)
    logging.info("Downloading Inference Data")
    logging.info("=" * 60)
    inference_df = download_inference_data()
    logging.info(f"Inference data shape: {inference_df.shape}")


if __name__ == '__main__':
    main()