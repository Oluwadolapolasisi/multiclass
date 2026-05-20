"""
train.py

Training script for the Human Activity Recognition multiclass classifier.
Loads the dataset, builds a FeedForwardNetwork, trains with TensorBoard
logging, and evaluates on the test set.

Usage:
    python train.py
    python train.py --data_dir ./data --epochs 100 --batch_size 64
    python train.py --evaluate_only --model_path saved_models/har_ffn

TensorBoard:
    tensorboard --logdir logs/tensorboard

Version: 1.0
Date: 02-05-2026
"""

import argparse
import os
import tensorflow as tf
from logger_setup import setup_logger, logger
from dataset import load_dataset, NUM_CLASSES
from models import build_feedforward_network, FeedForwardNetwork
from utils import check_gpu, set_gpu, set_seed, get_callbacks


def _resolve_model_path(path):
    """Append .keras extension if missing and the .keras file exists."""
    _, ext = os.path.splitext(path)
    if ext not in (".keras", ".h5"):
        keras_path = path + ".keras"
        if os.path.isfile(keras_path):
            logger.info(f"Resolved model path: {path} → {keras_path}")
            return keras_path
    return path


def train(args):
    """Run the full training pipeline."""

    set_gpu()
    check_gpu()
    set_seed(args.seed)

    train_path = os.path.join(args.data_dir, "train_dataset.csv")
    val_path = os.path.join(args.data_dir, "val_dataset.csv")
    test_path = os.path.join(args.data_dir, "test.csv")

    train_ds, train_info = load_dataset(
        train_path, batch_size=args.batch_size, evaluate_mode=False
    )
    val_ds, _ = load_dataset(
        val_path, batch_size=args.batch_size, evaluate_mode=True
    )

    num_features = train_info["num_features"]
    logger.info(
        f"Train samples: {train_info['num_samples']} | "
        f"Features: {num_features} | Classes: {NUM_CLASSES}"
    )

    model = build_feedforward_network(
        num_features=num_features,
        num_classes=NUM_CLASSES,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    callbacks = get_callbacks(
        log_dir=args.tensorboard_dir,
        patience=args.patience,
    )

    logger.info(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Save Model
    save_dir = os.path.dirname(args.model_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(args.model_path)
    logger.success(f"Model saved to {args.model_path}")

    # Evaluate on test set 
    if os.path.isfile(test_path):
        logger.info("Evaluating on test set...")
        test_ds, test_info = load_dataset(
            test_path, batch_size=args.batch_size, evaluate_mode=True
        )
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
        logger.success(
            f"Test results — loss: {test_loss:.4f} | accuracy: {test_accuracy:.4f}"
        )
    else:
        logger.warning(f"Test file not found at {test_path} — skipping evaluation.")

    return history


def evaluate(args):
    """Load a saved model and evaluate on the test set."""

    set_gpu()
    test_path = os.path.join(args.data_dir, "test.csv")

    if not os.path.isfile(test_path):
        logger.error(f"Test file not found: {test_path}")
        return

    logger.info(f"Loading saved model from {args.model_path}")
    model_path = _resolve_model_path(args.model_path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"FeedForwardNetwork": FeedForwardNetwork},
    )

    test_ds, test_info = load_dataset(
        test_path, batch_size=args.batch_size, evaluate_mode=True
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
    logger.success(
        f"Test results — loss: {test_loss:.4f} | accuracy: {test_accuracy:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a Feed-Forward Network for Human Activity Recognition."
    )

    # Data
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing the CSV data files.")

    # Model
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate after each hidden layer (default: 0.3).")
    parser.add_argument("--model_path", default="saved_models/har_ffn.keras",
                        help="Path to save/load the trained model (default: saved_models/har_ffn.keras).")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs (default: 100).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64).")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate for Adam (default: 0.001).")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    # Logging
    parser.add_argument("--tensorboard_dir", default="logs/tensorboard",
                        help="TensorBoard log directory.")

    # Mode
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Skip training; only evaluate a saved model on test set.")

    args = parser.parse_args()
    setup_logger()

    logger.info("=" * 60)
    logger.info("HAR Multiclass Classification — Feed-Forward Network")
    logger.info("=" * 60)

    if args.evaluate_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
