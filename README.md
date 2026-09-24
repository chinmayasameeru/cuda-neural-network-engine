# CUDA Neural Network Engine

A deep learning engine implemented in CUDA C++. The library provides GPU-accelerated tensor operations, automatic differentiation, and common neural network layers through a Python interface.

This project is a technical exploration of how GPU-accelerated neural network libraries work at the systems level.

---

## Architecture

The engine follows a layered design:

```
┌──────────────────────────────────────────┐
│            Python bindings (pybind11)      │
├──────────────────────────────────────────┤
│          Layer definitions                │
│   Linear, Conv2D, BatchNorm, Pooling,    │
│   Activations                            │
├──────────────────────────────────────────┤
│          Autograd engine                  │
│   Gradient computation graph             │
├──────────────────────────────────────────┤
│          CUDA kernels                     │
│   Tensor operations, convolution,        │
│   activation functions                   │
├──────────────────────────────────────────┤
│          cuBLAS                           │
│   Optimized matrix operations            │
├──────────────────────────────────────────┤
│          CUDA runtime                     │
│   Memory management, kernel launches     │
└──────────────────────────────────────────┘
```

Each layer uses only the primitives provided by the layer below it. The autograd engine records operations during forward pass execution and traverses the recorded graph in reverse to compute gradients. Matrix operations delegate to cuBLAS for performance.

---

## Implementation Status

### Tensor Operations

| Operation | Description | Status |
|-----------|-------------|--------|
| Allocation | GPU memory with shape metadata | ✅ |
| Transfer | Host ↔ Device copy | ✅ |
| Arithmetic | Add, multiply, scale | ✅ |
| Reduction | Sum along axis | ✅ |
| Reshape | View without copy | ✅ |

### Layers

| Layer | Description | Status |
|-------|-------------|--------|
| Linear | Fully connected with optional bias | ✅ |
| Conv2D | 2D convolution with configurable stride and padding | ✅ |
| BatchNorm2D | Batch normalization for convolutional networks | ✅ |
| MaxPool2D | 2D max pooling | ✅ |
| AdaptiveAvgPool2D | Adaptive average pooling to fixed output size | ✅ |
| ReLU | Rectified linear unit | ✅ |
| Sigmoid | Logistic activation | ✅ |
| Tanh | Hyperbolic tangent activation | ✅ |
| LeakyReLU | Leaky rectified linear unit | ✅ |
| GELU | Gaussian error linear unit | ✅ |
| SiLU | Sigmoid linear unit (Swish) | ✅ |
| Softmax | Softmax activation | ✅ |
| Dropout | Random unit masking during training | ✅ |
| Flatten | Spatial flattening for transition to dense layers | ✅ |

### Optimizers

| Optimizer | Description | Status |
|-----------|-------------|--------|
| SGD | Stochastic gradient descent with momentum and weight decay | ✅ |
| Adam | Adaptive moment estimation with bias correction | ✅ |

### Loss Functions

| Loss | Description | Status |
|------|-------------|--------|
| Cross Entropy | Softmax + negative log-likelihood | ✅ |
| MSE | Mean squared error | ✅ |

---

## Requirements

- NVIDIA GPU with Compute Capability ≥ 6.0
- CUDA Toolkit 12.x
- C++17 compatible compiler
- Python 3.10+
- pybind11
- numpy

---

## Build

```bash
git clone https://github.com/chinmayasameeru/cuda-neural-network-engine.git
cd cuda-neural-network-engine
pip install pybind11 numpy
python setup.py build_ext --inplace
```

---

## Usage

```python
import cnn_engine as cnn
import numpy as np

model = cnn.Sequential()
model.add(cnn.Linear(784, 256))
model.add(cnn.ReLU())
model.add(cnn.Linear(256, 128))
model.add(cnn.ReLU())
model.add(cnn.Linear(128, 10))

X = np.random.randn(64, 784).astype(np.float32)
y = np.zeros((64, 10), dtype=np.float32)
y[np.arange(64), np.random.randint(0, 10, 64)] = 1.0

optimizer = cnn.Adam(model.parameters(), lr=0.001)
loss_fn = cnn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
```

---

## Project Structure

```
cuda-neural-network-engine/
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── bindings/
│   └── python_bindings.cpp
├── docs/
│   ├── autograd.md
│   └── memory.md
├── include/
│   ├── tensor.h
│   ├── kernels.cuh
│   ├── layers.h
│   └── model.h
├── src/
│   ├── tensor.cpp
│   ├── kernels.cu
│   ├── layers.cpp
│   └── optimizers.cpp
├── tests/
│   └── test_engine.py
├── benchmarks/
│   └── benchmark.py
├── setup.py
├── Makefile
├── .gitignore
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
└── README.md
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
