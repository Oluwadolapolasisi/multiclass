"""
dataset.py

"""

import tensorflow as tf


def load_dataset(data_path, evaluate_mode=False):
    DATASET = tf.data.Dataset
