#!/usr/bin/env python3
"""
CUDA Neural Network Engine — Test Suite
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import cnn_engine as cnn

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


def test_sequential():
    model = cnn.Sequential()
    model.add("Linear", [784, 256, 1])
    model.add("ReLU")
    model.add("Linear", [256, 10, 1])
    s = model.summary()
    assert "Sequential" in s
    assert "Linear" in s
    assert "ReLU" in s


def test_forward():
    model = cnn.Sequential()
    model.add("Linear", [10, 5, 1])
    model.add("ReLU")
    X = np.random.randn(4, 10).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (4, 5)


def test_conv():
    model = cnn.Sequential()
    model.add("Conv2D", [3, 16, 3, 1, 1, 1])
    model.add("ReLU")
    X = np.random.randn(2, 3, 32, 32).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (2, 16, 32, 32)


def test_pool():
    model = cnn.Sequential()
    model.add("MaxPool2D", [2, 2])
    X = np.random.randn(2, 3, 8, 8).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (2, 3, 4, 4)


def test_batchnorm():
    model = cnn.Sequential()
    model.add("BatchNorm2D", [16])
    X = np.random.randn(2, 16, 8, 8).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (2, 16, 8, 8)


def test_flatten():
    model = cnn.Sequential()
    model.add("Flatten")
    X = np.random.randn(2, 3, 4, 4).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (2, 48)


def test_dropout():
    model = cnn.Sequential()
    model.add("Dropout", [0.5])
    X = np.random.randn(2, 10).astype(np.float32)
    model.train()
    y = model.forward(X)
    assert y.shape == (2, 10)


def test_activations():
    model_relu = cnn.Sequential()
    model_relu.add("ReLU")
    X = np.array([[-1, 0, 1, 2]], dtype=np.float32)
    y = model_relu.forward(X)
    assert y.shape == (1, 4)

    model_sig = cnn.Sequential()
    model_sig.add("Sigmoid")
    y = model_sig.forward(X)
    assert np.all(y >= 0) and np.all(y <= 1)


def test_optimizers():
    sgd = cnn.SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
    adam = cnn.Adam(lr=0.001)
    assert sgd is not None
    assert adam is not None


def test_losses():
    ce = cnn.CrossEntropyLoss()
    mse = cnn.MSELoss()
    assert ce is not None
    assert mse is not None


def test_mnist_loop():
    model = cnn.Sequential()
    model.add("Linear", [784, 128, 1])
    model.add("ReLU")
    model.add("Linear", [128, 10, 1])

    X = np.random.randn(32, 784).astype(np.float32)
    y = np.zeros((32, 10), dtype=np.float32)
    y[np.arange(32), np.random.randint(0, 10, 32)] = 1.0

    logits = model.forward(X)
    assert logits.shape == (32, 10)


if __name__ == "__main__":
    print("Running CUDA Neural Network Engine tests...\n")
    test_sequential()
    test_forward()
    test_conv()
    test_pool()
    test_batchnorm()
    test_flatten()
    test_dropout()
    test_activations()
    test_optimizers()
    test_losses()
    test_mnist_loop()
    print("\nAll tests passed!")
