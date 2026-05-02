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

import kagglehub
from logger_setup import setup_logger, logger


def download_dataset():
    try:
        kagglehub.dataset_download(
            'uciml/human-activity-recognition-with-smartphones',  output_dir='./data')
    except Exception as error:
        logger.error(f"Error occurred while downloading dataset: {error}")


if __name__ == "__main__":
    setup_logger()
    logger.info("Starting dataset download...")
    download_dataset()
    logger.success("Dataset download is completed.")
