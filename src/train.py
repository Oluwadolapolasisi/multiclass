"""
train.py

Training script for the Human Activity Recognition multiclass classifier.
Supports both FeedForwardNetwork (ffn) and Conv1DNetwork (cnn) architectures.
Loads the dataset, builds the selected model, trains with TensorBoard
logging, computes evaluation metrics (confusion matrix, F1, precision,
recall, AUC), and evaluates on the test set.

Usage:
    python train.py --model ffn --data_dir ./data --epochs 100
    python train.py --model cnn --data_dir ./data --epochs 100
    python train.py --evaluate_only --model ffn --model_path saved_models/har_ffn.keras

TensorBoard:
    tensorboard --logdir logs/tensorboard

Version: 2.0
Date: 20-05-2026
"""

import argparse
import os
import tensorflow as tf
from logger_setup import setup_logger, logger
from dataset import load_dataset, NUM_CLASSES, ACTIVITY_LABELS
from models import (
    build_model,
    FeedForwardNetwork,
    Conv1DNetwork,
    MODEL_REGISTRY,
)
from metrics import compute_all_metrics, log_metrics_to_tensorboard
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

    model = build_model(
        model_type=args.model,
        num_features=num_features,
        num_classes=NUM_CLASSES,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    # Namespace TensorBoard logs by model type
    tb_dir = os.path.join(args.tensorboard_dir, args.model)
    callbacks = get_callbacks(
        log_dir=tb_dir,
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

    # Evaluate on test set + compute advanced metrics
    if os.path.isfile(test_path):
        logger.info("Evaluating on test set...")
        test_ds, test_info = load_dataset(
            test_path, batch_size=args.batch_size, evaluate_mode=True
        )
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
        logger.success(
            f"Test results — loss: {test_loss:.4f} | accuracy: {test_accuracy:.4f}"
        )

        # Compute and save advanced metrics
        metrics_dir = os.path.join("logs", "metrics", args.model)
        metrics = compute_all_metrics(
            model, test_ds,
            class_names=ACTIVITY_LABELS,
            output_dir=metrics_dir,
        )
        log_metrics_to_tensorboard(metrics, log_dir=tb_dir)
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
        custom_objects={
            "FeedForwardNetwork": FeedForwardNetwork,
            "Conv1DNetwork": Conv1DNetwork,
        },
    )

    test_ds, test_info = load_dataset(
        test_path, batch_size=args.batch_size, evaluate_mode=True
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
    logger.success(
        f"Test results — loss: {test_loss:.4f} | accuracy: {test_accuracy:.4f}"
    )

    # Compute and save advanced metrics
    metrics_dir = os.path.join("logs", "metrics", args.model)
    metrics = compute_all_metrics(
        model, test_ds,
        class_names=ACTIVITY_LABELS,
        output_dir=metrics_dir,
    )

    # Log to TensorBoard
    tb_dir = os.path.join(args.tensorboard_dir, args.model)
    log_metrics_to_tensorboard(metrics, log_dir=tb_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a HAR model (FFN or CNN)."
    )

    # Model selection
    parser.add_argument("--model", default="ffn", choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to use: 'ffn' or 'cnn' (default: ffn).")

    # Data
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing the CSV data files.")

    # Model saving/loading
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate after each hidden layer (default: 0.3).")
    parser.add_argument("--model_path", default=None,
                        help="Path to save/load the trained model. "
                             "Defaults to saved_models/har_{model}.keras.")

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

    # Default model_path based on model type if not explicitly provided
    if args.model_path is None:
        args.model_path = f"saved_models/har_{args.model}.keras"

    setup_logger()

    model_name = "Feed-Forward Network" if args.model == "ffn" else "1D-CNN"
    logger.info("=" * 60)
    logger.info(f"HAR Multiclass Classification — {model_name}")
    logger.info("=" * 60)

    if args.evaluate_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
