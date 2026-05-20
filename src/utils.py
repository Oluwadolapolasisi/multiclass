"""
utils.py

Utility functions for the HAR multiclass classification project.
Handles GPU configuration, reproducibility, and common helpers.

Version: 1.0
Date: 02-05-2026
"""

import os
import random
import numpy as np
import tensorflow as tf
from logger_setup import logger


def check_gpu():
    """Log available GPU devices and return the count.

    Returns:
        int: Number of GPUs detected.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            logger.info(f"GPU detected: {gpu.name}")
    else:
        logger.warning("No GPU detected — training will run on CPU.")
    return len(gpus)


def set_gpu(memory_growth=True):
    """Configure GPU settings for TensorFlow.

    Enables memory growth by default so TF doesn't pre-allocate the
    entire GPU memory, which is helpful when sharing the GPU or when
    running alongside other processes.

    Args:
        memory_growth: If True, allow dynamic GPU memory allocation.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled.")
        except RuntimeError as e:
            logger.error(f"Failed to set GPU memory growth: {e}")


def set_seed(seed=42):
    """Set random seeds across Python, NumPy, and TensorFlow for
    reproducibility.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seed set to {seed}")


def get_callbacks(log_dir="logs/tensorboard", patience=10):
    """Return a standard set of Keras callbacks for training.

    Args:
        log_dir:  Directory for TensorBoard log files.
        patience: Number of epochs with no improvement before
                  early stopping triggers.

    Returns:
        List of tf.keras.callbacks.
    """
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    logger.info(
        f"Callbacks: TensorBoard (log_dir={log_dir}), "
        f"EarlyStopping (patience={patience}), "
        f"ReduceLROnPlateau"
    )
    return callbacks
