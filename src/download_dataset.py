"""
download_dataset.py

This module is responsible for downloading the Human Activity Recognition dataset from Kaggle.
It uses the kagglehub library to access the Kaggle API and retrieve the specified dataset.
The dataset is downloaded to the './data' directory, which can be used for further processing and analysis in the application.
This script can be executed independently to ensure that the dataset is available locally for training and evaluation purposes.

Usage example:
    python download_dataset.py

Functions:
    download_dataset(): Downloads the specified dataset from Kaggle and saves it to the local directory.

Exceptions:
    Any exceptions that occur during the dataset download process are logged using the Loguru logger for debugging
    and error tracking.

Version: 1.0
Date: 02-05-2026
"""

import argparse
import os
import kagglehub
import pandas as pd
from logger_setup import setup_logger, logger
from sklearn.model_selection import train_test_split


def download_dataset(data_path=None):
    try:
        if data_path is None:
            logger.warning(
                "Invalid Data Path given. Setting default data path as './data'")
            data_path = './data'
        kagglehub.dataset_download(
            'uciml/human-activity-recognition-with-smartphones',  output_dir=data_path)
    except Exception as error:
        logger.error(f"Error occurred while downloading dataset: {error}")


def split_dataset(file_path, data_dir, val_size=0.15):
    logger.info(f"Reading dataset from: {file_path}")
    dataframe = pd.read_csv(file_path)
    logger.info(f"Dataset shape: {dataframe.shape}")

    train_data, val_data = train_test_split(
        dataframe, test_size=val_size, shuffle=False)

    train_path = os.path.join(data_dir, 'train_dataset.csv')
    val_path = os.path.join(data_dir, 'val_dataset.csv')

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)

    logger.info(f"Train set: {train_data.shape[0]} samples saved to {train_path}")
    logger.info(f"Validation set: {val_data.shape[0]} samples saved to {val_path}")
    return train_path, val_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the Human Activity Recognition dataset from Kaggle."
    )
    parser.add_argument("--data_path", default="./data",
                        help="Path to save the dataset (default: ./data)")
    parser.add_argument("--split_data", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Split the data into train and validation set (default: True)")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fraction of data to use for validation (default: 0.15)")
    args = parser.parse_args()
    setup_logger()
    logger.info("Starting dataset download...")
    try:
        download_dataset(args.data_path)
        logger.success("Dataset download completed.")
        if args.split_data:
            logger.info("Splitting the data into train and validation set...")
            file_path = os.path.join(args.data_path, 'train.csv')
            train_path, val_path = split_dataset(
                file_path, data_dir=args.data_path, val_size=args.val_size
            )
            logger.success("Data split completed successfully.")
    except Exception as error:
        logger.error(f"Error occurred while downloading dataset: {error}")
