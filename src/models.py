"""
model.py


Layer 1: 512 neurons
Layer 2: 256 neurons
Layer 3: 128 neurons

Dropout (rate of 0.3 to 0.5) after each hidden layer. This forces the network to learn redundant representations and not rely on specific neurons.
Batch Normalization after each hidden layer, before the activation. It stabilizes training, allows higher learning rates, and has a mild regularizing effect of its own.
Batch size: 32 to 64

"""

import tensorflow as tf


#TODO CNN, FeedFN

class FeedForwardNetwork():
    def __init__(self, ):
        

