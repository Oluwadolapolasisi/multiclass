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

Architecture — Conv1DNetwork
    Treats the 561 pre-computed features as a 1D signal and applies
    1D convolutions to discover local correlations among groups of
    related features.

    Convolutional blocks:
        Conv1D → BatchNorm → ReLU → MaxPool1D → Dropout

    Final classification head:
        GlobalAveragePooling1D → Dense(128) → BatchNorm → ReLU → Dropout → Dense(6, softmax)

Version: 2.0
Date: 20-05-2026
"""

import keras
import tensorflow as tf
from logger_setup import logger


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

        # Build hidden blocks
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

@keras.saving.register_keras_serializable()
class Conv1DNetwork(tf.keras.Model):
    """1D-CNN for classification on pre-computed feature vectors.

    Reshapes the flat 561-feature input into (561, 1) and applies 1D
    convolutions to capture local feature correlations.

    Architecture:
        Reshape(561, 1)
        → Conv1D(64,  k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
        → Conv1D(128, k=5) → BN → ReLU → MaxPool(2) → Dropout(0.3)
        → Conv1D(256, k=3) → BN → ReLU → GlobalAvgPool1D
        → Dense(128) → BN → ReLU → Dropout(0.5)
        → Dense(num_classes, softmax)

    Args:
        num_features: Number of input features (e.g. 561).
        num_classes:  Number of output classes (e.g. 6).
        dropout_rate: Dropout probability for conv blocks (default 0.3).
    """

    def __init__(
        self,
        num_features,
        num_classes,
        dropout_rate=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_features = num_features
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate

        # Reshape flat features to (561, 1) for Conv1D
        self.reshape = tf.keras.layers.Reshape((num_features, 1))

        # Conv block 1
        self.conv1 = tf.keras.layers.Conv1D(
            64, kernel_size=5, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        # Conv block 2
        self.conv2 = tf.keras.layers.Conv1D(
            128, kernel_size=5, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        # Conv block 3
        self.conv3 = tf.keras.layers.Conv1D(
            256, kernel_size=3, padding="same", use_bias=False
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        self.gap = tf.keras.layers.GlobalAveragePooling1D()

        # Classification head
        self.dense1 = tf.keras.layers.Dense(128, use_bias=False)
        self.bn_fc = tf.keras.layers.BatchNormalization()
        self.relu_fc = tf.keras.layers.ReLU()
        self.drop_fc = tf.keras.layers.Dropout(0.5)

        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation="softmax"
        )

    def call(self, inputs, training=False):
        x = self.reshape(inputs)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        x = self.gap(x)

        # Classification head
        x = self.dense1(x)
        x = self.bn_fc(x, training=training)
        x = self.relu_fc(x)
        x = self.drop_fc(x, training=training)

        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self._num_features,
            "num_classes": self._num_classes,
            "dropout_rate": self._dropout_rate,
        })
        return config

    def summary_str(self):
        """Return a human-readable summary of the architecture."""
        return (
            f"Conv1DNetwork | "
            f"filters=[64, 128, 256] | "
            f"fc=128 | "
            f"dropout={self._dropout_rate}"
        )


def build_conv1d_network(
    num_features,
    num_classes,
    dropout_rate=0.3,
    learning_rate=1e-3,
):
    """Construct, compile, and return a Conv1DNetwork.

    Args:
        num_features:  Number of input features.
        num_classes:   Number of target classes.
        dropout_rate:  Dropout probability for conv blocks.
        learning_rate: Initial learning rate for Adam.

    Returns:
        A compiled tf.keras.Model ready for .fit().
    """
    model = Conv1DNetwork(
        num_features=num_features,
        num_classes=num_classes,
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


# Registry of available model types
MODEL_REGISTRY = {
    "ffn": build_feedforward_network,
    "cnn": build_conv1d_network,
}


def build_model(model_type, num_features, num_classes, dropout_rate=0.3, learning_rate=1e-3):
    """Build and compile a model by type name.

    Args:
        model_type:    One of 'ffn' or 'cnn'.
        num_features:  Number of input features.
        num_classes:   Number of target classes.
        dropout_rate:  Dropout probability.
        learning_rate: Initial learning rate for Adam.

    Returns:
        A compiled tf.keras.Model.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    builder = MODEL_REGISTRY[model_type]
    return builder(
        num_features=num_features,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
