"""
models.py

Neural network architectures for Human Activity Recognition (HAR)
multiclass classification (6 activity classes, 561 input features).

Architecture — FeedForwardNetwork
    Hidden layers:
        Layer 1: 512 neurons
        Layer 2: 256 neurons
        Layer 3: 128 neurons

    Each hidden block follows the pattern:
        Dense → BatchNormalization → ReLU → Dropout

    Batch Normalization is placed before the activation. It stabilises
    training, permits higher learning rates, and provides a mild
    regularising effect.

    Dropout (default 0.3) after each hidden layer forces the network
    to learn redundant representations and not rely on specific neurons.

    Output layer: Dense(num_classes) with softmax activation for
    multiclass probability distribution.

Version: 1.0
Date: 20-05-2026
"""

import keras
import tensorflow as tf
from logger_setup import logger


# TODO: CNN architecture


@keras.saving.register_keras_serializable()
class FeedForwardNetwork(tf.keras.Model):
    """Fully-connected feedforward network for classification.

    Args:
        num_features: Number of input features (e.g. 561).
        num_classes:  Number of output classes (e.g. 6).
        hidden_units: Sequence of hidden layer widths.
        dropout_rate: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_features,
        num_classes,
        hidden_units=(512, 256, 128),
        dropout_rate=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store config for serialization (get_config)
        self._num_features = num_features
        self._num_classes = num_classes
        self._hidden_units = tuple(hidden_units)
        self._dropout_rate = dropout_rate

        # Build hidden blocks: Dense → BatchNorm → ReLU → Dropout
        self.hidden_blocks = []
        for units in hidden_units:
            block = [
                tf.keras.layers.Dense(units, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(dropout_rate),
            ]
            self.hidden_blocks.append(block)

        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation="softmax"
        )

    def call(self, inputs, training=False):
        x = inputs
        for block in self.hidden_blocks:
            for layer in block:
                # BatchNorm and Dropout behave differently during training
                if isinstance(
                    layer,
                    (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout),
                ):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self._num_features,
            "num_classes": self._num_classes,
            "hidden_units": self._hidden_units,
            "dropout_rate": self._dropout_rate,
        })
        return config

    def summary_str(self):
        """Return a human-readable summary of the architecture."""
        hidden = [b[0].units for b in self.hidden_blocks]
        return (
            f"FeedForwardNetwork | "
            f"hidden={hidden} | "
            f"dropout={self.hidden_blocks[0][3].rate}"
        )


def build_feedforward_network(
    num_features,
    num_classes,
    hidden_units=(512, 256, 128),
    dropout_rate=0.3,
    learning_rate=1e-3,
):
    """Construct, compile, and return a FeedForwardNetwork.

    Uses Adam optimiser with sparse categorical cross-entropy loss
    (suitable for integer-encoded labels produced by dataset.py).

    Args:
        num_features:  Number of input features.
        num_classes:   Number of target classes.
        hidden_units:  Tuple of hidden layer widths.
        dropout_rate:  Dropout probability after each hidden layer.
        learning_rate: Initial learning rate for Adam.

    Returns:
        A compiled tf.keras.Model ready for .fit().
    """
    model = FeedForwardNetwork(
        num_features=num_features,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Trigger weight creation so .summary() works immediately
    model.build(input_shape=(None, num_features))

    logger.info(f"Model built — {model.summary_str()}")
    model.summary(print_fn=logger.info)

    return model
