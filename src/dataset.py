"""
dataset.py

Loads the Human Activity Recognition dataset from CSV files and constructs
tf.data.Dataset pipelines for training, validation, and evaluation.

The module handles:
    - Reading CSV files produced by download_dataset.py (train_dataset.csv,
      val_dataset.csv) or the original Kaggle files (train.csv, test.csv).
    - Encoding the string 'Activity' labels into integer class indices.
    - Dropping non-feature columns ('subject') that should not be used as
      model inputs.
    - Building efficient tf.data.Dataset pipelines with proper shuffling,
      batching, and prefetching.

Usage:
    from dataset import load_dataset, ACTIVITY_LABELS

    train_ds, info = load_dataset("./data/train_dataset.csv", batch_size=64)
    val_ds, _     = load_dataset("./data/val_dataset.csv", batch_size=64, evaluate_mode=True)

Version: 1.0
Date: 02-05-2026
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from logger_setup import logger


# Ordering of the 6 activity classes.
# Index position becomes the integer label (0-5).
ACTIVITY_LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

NUM_CLASSES = len(ACTIVITY_LABELS)

# Columns that are present in the CSV but should NOT be used as input features.
_NON_FEATURE_COLUMNS = {"subject", "Activity"}


def _encode_labels(activity_series):
    """Map string activity names to integer class indices.

    Args:
        activity_series: A pandas Series of string activity labels.

    Returns:
        A numpy array of int32 class indices.

    Raises:
        ValueError: If an unknown activity label is encountered.
    """
    label_to_index = {label: idx for idx, label in enumerate(ACTIVITY_LABELS)}
    try:
        encoded = activity_series.map(label_to_index).to_numpy(dtype=np.int32)
    except Exception as exc:
        unknown = set(activity_series.unique()) - set(ACTIVITY_LABELS)
        raise ValueError(
            f"Unknown activity labels encountered: {unknown}"
        ) from exc

    # Catch any NaNs that slipped through
    if np.any(np.isnan(activity_series.map(label_to_index))):
        unknown = set(activity_series.unique()) - set(ACTIVITY_LABELS)
        raise ValueError(f"Unknown activity labels encountered: {unknown}")

    return encoded


def load_dataset(data_path, batch_size=64, evaluate_mode=False, shuffle_buffer=10000):
    """Load a HAR CSV file and return a batched tf.data.Dataset.

    Args:
        data_path:      Path to a CSV file (e.g. train_dataset.csv, test.csv).
        batch_size:     Number of samples per batch.
        evaluate_mode:  If True, disables shuffling and repeating so the
                        dataset is iterated exactly once (for validation /
                        test evaluation).
        shuffle_buffer: Size of the shuffle buffer used during training.

    Returns:
        A tuple (dataset, info) where:
            dataset: A tf.data.Dataset yielding (features, labels) batches.
                     features shape: (batch_size, num_features)
                     labels shape:   (batch_size,)  — integer class indices
            info:    A dict with metadata:
                     - "num_samples":  Total number of rows in the CSV.
                     - "num_features": Number of input features.
                     - "num_classes":  Number of activity classes (6).
                     - "activity_labels": List of class name strings.
                     - "steps_per_epoch": Number of batches per full pass.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError:        If required columns are missing.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    logger.info(f"Loading dataset from {data_path}")
    dataframe = pd.read_csv(data_path)
    logger.info(f"Loaded {len(dataframe)} samples with {len(dataframe.columns)} columns")

    if "Activity" not in dataframe.columns:
        raise ValueError(
            f"'Activity' column not found in {data_path}. "
            f"Available columns: {list(dataframe.columns[:10])}..."
        )

    # Separate features and labels
    feature_columns = [
        col for col in dataframe.columns if col not in _NON_FEATURE_COLUMNS
    ]
    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    labels = _encode_labels(dataframe["Activity"])

    num_samples = features.shape[0]
    num_features = features.shape[1]
    steps_per_epoch = int(np.ceil(num_samples / batch_size))

    logger.info(
        f"Features shape: {features.shape} | "
        f"Labels shape: {labels.shape} | "
        f"Classes: {NUM_CLASSES}"
    )

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if not evaluate_mode:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    info = {
        "num_samples": num_samples,
        "num_features": num_features,
        "num_classes": NUM_CLASSES,
        "activity_labels": ACTIVITY_LABELS,
        "steps_per_epoch": steps_per_epoch,
    }

    logger.info(
        f"tf.data.Dataset ready — "
        f"{'EVAL' if evaluate_mode else 'TRAIN'} mode | "
        f"{steps_per_epoch} steps/epoch | batch_size={batch_size}"
    )

    return dataset, info
