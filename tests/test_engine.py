#!/usr/bin/env python3
"""
CUDA Neural Network Engine — Test Suite v0.2.0
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
    print("cnn_engine not available — build with: python setup.py build_ext --inplace")


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


# ============================================================
# v0.2.0 Tests - Advanced Layers
# ============================================================


def test_residual_block():
    """Test ResNet-style residual block."""
    model = cnn.Sequential()
    model.add("Conv2D", [3, 64, 3, 1, 1, 1])
    model.add("BatchNorm2D", [64])
    model.add("ReLU")
    model.add("ResidualBlock", [64, 64, 1])
    model.add("AdaptiveAvgPool2D", [1, 1])
    model.add("Flatten")
    model.add("Linear", [64, 10, 1])

    X = np.random.randn(2, 3, 32, 32).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (2, 10)


def test_lstm_cell():
    """Test LSTM for sequence processing."""
    model = cnn.Sequential()
    model.add("LSTMCell", [10, 20])

    # Single timestep
    X = np.random.randn(1, 10).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (1, 20)


def test_gru_cell():
    """Test GRU for sequence processing."""
    model = cnn.Sequential()
    model.add("GRUCell", [10, 20])

    X = np.random.randn(1, 10).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (1, 20)


def test_layer_norm():
    """Test layer normalization."""
    model = cnn.Sequential()
    model.add("Linear", [784, 256])
    model.add("LayerNorm", [256])
    model.add("ReLU")

    X = np.random.randn(4, 784).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (4, 256)


def test_sequence_to_sequence():
    """Test LSTM over multiple timesteps."""
    model = cnn.Sequential()
    model.add("LSTMCell", [10, 20])

    # Process sequence of length 5
    for t in range(5):
        X_t = np.random.randn(1, 10).astype(np.float32)
        y_t = model.forward(X_t)
        assert y_t.shape == (1, 20)


# ============================================================
# v0.2.0 Tests - Schedulers
# ============================================================


def test_step_lr():
    """Test step learning rate scheduler."""
    scheduler = cnn.StepLR(initial_lr=0.1, step_size=10, gamma=0.1)
    assert abs(scheduler.get_lr(0) - 0.1) < 1e-6
    assert abs(scheduler.get_lr(10) - 0.01) < 1e-6
    assert abs(scheduler.get_lr(20) - 0.001) < 1e-6


def test_cosine_annealing():
    """Test cosine annealing LR scheduler."""
    scheduler = cnn.CosineAnnealingLR(initial_lr=0.1, T_max=100, eta_min=0.001)

    # Start at initial lr
    assert abs(scheduler.get_lr(0) - 0.1) < 1e-6

    # End at eta_min
    assert abs(scheduler.get_lr(100) - 0.001) < 1e-6

    # Middle should be between
    mid_lr = scheduler.get_lr(50)
    assert 0.001 < mid_lr < 0.1


def test_resnet_style_model():
    """Test a complete ResNet-style architecture."""
    model = cnn.Sequential()

    # Stem
    model.add("Conv2D", [3, 64, 7, 2, 3, 1])
    model.add("BatchNorm2D", [64])
    model.add("ReLU")
    model.add("MaxPool2D", [3, 2])

    # Residual blocks
    model.add("ResidualBlock", [64, 128, 2])
    model.add("ResidualBlock", [128, 256, 2])
    model.add("ResidualBlock", [256, 512, 2])

    # Classifier
    model.add("AdaptiveAvgPool2D", [1, 1])
    model.add("Flatten")
    model.add("Linear", [512, 1000, 1])

    X = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y = model.forward(X)
    assert y.shape == (1, 1000)


if __name__ == "__main__":
    print("Running CUDA Neural Network Engine v0.2.0 tests...\n")

    # v0.1.0 tests
    print("v0.1.0 Core Tests:")
    test_sequential()
    print("  PASS: Sequential")
    test_forward()
    print("  PASS: Forward")
    test_conv()
    print("  PASS: Conv2D")
    test_pool()
    print("  PASS: MaxPool2D")
    test_batchnorm()
    print("  PASS: BatchNorm2D")
    test_flatten()
    print("  PASS: Flatten")
    test_dropout()
    print("  PASS: Dropout")
    test_activations()
    print("  PASS: Activations")
    test_optimizers()
    print("  PASS: Optimizers")
    test_losses()
    print("  PASS: Losses")
    test_mnist_loop()
    print("  PASS: MNIST loop")

    # v0.2.0 tests
    print("\nv0.2.0 Advanced Layer Tests:")
    test_residual_block()
    print("  PASS: ResidualBlock")
    test_lstm_cell()
    print("  PASS: LSTMCell")
    test_gru_cell()
    print("  PASS: GRUCell")
    test_layer_norm()
    print("  PASS: LayerNorm")
    test_sequence_to_sequence()
    print("  PASS: Sequence processing")
    test_resnet_style_model()
    print("  PASS: ResNet-style model")

    print("\nv0.2.0 Scheduler Tests:")
    test_step_lr()
    print("  PASS: StepLR")
    test_cosine_annealing()
    print("  PASS: CosineAnnealingLR")

    print("\n" + "=" * 50)
    print("All v0.2.0 tests passed!")
    print("=" * 50)
