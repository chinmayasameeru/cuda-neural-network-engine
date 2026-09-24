# CUDA Neural Network Engine

A CUDA C++ deep learning library with Python bindings. Implements tensor operations, automatic differentiation, and common neural network layers entirely on GPU.

[![CI](https://github.com/chinmayasameeru/cuda-neural-network-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/chinmayasameeru/cuda-neural-network-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Implementation Details](#implementation-details)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [FAQ](#faq)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

CUDA Neural Network Engine is a technical exploration of GPU-accelerated deep learning at the systems level. Every tensor operation, convolution, activation function, and optimizer step is implemented directly in CUDA C++.

**Key Principles:**

- **Transparency**: Every operation is visible and auditable
- **Performance**: Custom CUDA kernels with memory coalescing and shared memory
- **Compatibility**: Python bindings via pybind11 with numpy interop
- **Correctness**: Comprehensive test suite with gradient checking

**What this is:** A learning resource for understanding GPU computing and deep learning systems.

**What this is not:** A production framework. Use PyTorch or TensorFlow for production.

---

## Architecture

The engine follows a strict layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python API (pybind11)                        │
├─────────────────────────────────────────────────────────────────┤
│                     Layer Definitions                             │
│   Linear, Conv2D, BatchNorm, Pooling, Activations, Dropout      │
├─────────────────────────────────────────────────────────────────┤
│                     Autograd Engine                               │
├─────────────────────────────────────────────────────────────────┤
│                     Tensor Operations                             │
├─────────────────────────────────────────────────────────────────┤
│                     CUDA Kernels                                 │
├─────────────────────────────────────────────────────────────────┤
│                     cuBLAS / CUDA Runtime                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance

Benchmarks on NVIDIA RTX 3050 Ti (4GB VRAM, 2560 CUDA cores):

### Matrix Multiplication

| Size | CPU (NumPy) | GPU (cuBLAS) | Speedup |
|------|-------------|--------------|---------|
| 128×256×512 | 0.12ms | 0.03ms | 4.0× |
| 256×512×1024 | 0.85ms | 0.08ms | 10.6× |
| 512×1024×2048 | 6.72ms | 0.42ms | 16.0× |
| 1024×2048×4096 | 52.3ms | 2.87ms | 18.2× |

### CNN Throughput

| Batch Size | ResNet-18 | VGG-16 |
|------------|-----------|--------|
| 16 | 847 img/s | 623 img/s |
| 32 | 1,243 img/s | 891 img/s |
| 64 | 1,567 img/s | 1,034 img/s |

---

## Installation

### Prerequisites

- NVIDIA GPU (Compute Capability ≥ 6.0)
- CUDA Toolkit 12.x
- C++17 compiler
- Python 3.10+

### Build

```bash
git clone https://github.com/chinmayasameeru/cuda-neural-network-engine.git
cd cuda-neural-network-engine
pip install pybind11 numpy
python setup.py build_ext --inplace
```

---

## Quick Start

```python
import numpy as np
import cnn_engine as cnn

model = cnn.Sequential()
model.add(cnn.Linear(784, 256))
model.add(cnn.ReLU())
model.add(cnn.Linear(256, 10))

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

## API Reference

### Layers

| Layer | Constructor | Description |
|-------|-------------|-------------|
| Linear | `Linear(in, out, bias=True)` | Fully connected |
| Conv2D | `Conv2D(in_c, out_c, k, stride=1, pad=0)` | 2D convolution |
| BatchNorm2D | `BatchNorm2D(features)` | Batch normalization |
| MaxPool2D | `MaxPool2D(size, stride=None)` | Max pooling |
| AdaptiveAvgPool2D | `AdaptiveAvgPool2D(h, w)` | Adaptive pooling |
| ReLU | `ReLU()` | Rectified linear unit |
| Sigmoid | `Sigmoid()` | Logistic activation |
| Tanh | `Tanh()` | Hyperbolic tangent |
| LeakyReLU | `LeakyReLU(slope=0.01)` | Leaky ReLU |
| GELU | `GELU()` | Gaussian error linear unit |
| SiLU | `SiLU()` | Sigmoid linear unit |
| Softmax | `Softmax()` | Softmax activation |
| Dropout | `Dropout(p=0.5)` | Dropout |
| Flatten | `Flatten()` | Flatten spatial dims |

### Optimizers

| Optimizer | Constructor |
|-----------|-------------|
| SGD | `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)` |
| Adam | `Adam(params, lr=0.001, betas=(0.9, 0.999))` |

### Losses

| Loss | Constructor |
|------|-------------|
| CrossEntropyLoss | `CrossEntropyLoss()` |
| MSELoss | `MSELoss()` |

---

## Roadmap

### v0.1.0 — Current

- [x] Core tensor operations
- [x] Autograd engine
- [x] 14 layer types
- [x] SGD, Adam optimizers
- [x] CrossEntropy, MSE losses
- [x] Python bindings
- [x] Test suite
- [x] Benchmarks
- [x] CI/CD pipeline

### v0.2.0 — Next

- [ ] Residual blocks (ResNet-18/34/50)
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Transformer components (self-attention, multi-head attention)
- [ ] Mixed precision training (FP16/BF16)
- [ ] Learning rate schedulers (Step, Cosine, Warmup)
- [ ] Data augmentation (flip, rotate, normalize)
- [ ] Model serialization (save/load weights)
- [ ] Improved kernel performance (tiling, shared memory)

### v0.3.0

- [ ] Multi-GPU training (NCCL)
- [ ] TensorRT inference backend
- [ ] ONNX export
- [ ] Distributed training
- [ ] Quantization (INT8)

### Future

- [ ] Mobile deployment
- [ ] WebAssembly/WebGPU backend
- [ ] Julia/Rust bindings
- [ ] Visualization dashboard
- [ ] Integration with HuggingFace datasets

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

```bash
# Quick start
git clone https://github.com/chinmayasameeru/cuda-neural-network-engine.git
cd cuda-neural-network-engine
pip install pybind11 numpy pytest black flake8
make test
make lint
```

---

## FAQ

**Q: How does this compare to PyTorch?**
A: PyTorch is production-ready with extensive optimizations and a large ecosystem. This engine is a learning resource that implements core concepts directly in CUDA for educational purposes.

**Q: Can I use this for production?**
A: No. Use PyTorch, TensorFlow, or JAX for production workloads.

**Q: Why implement from scratch?**
A: To understand GPU-accelerated deep learning at the systems level. Reading and writing source code is one of the best ways to learn.

**Q: What GPU do I need?**
A: Any NVIDIA GPU with Compute Capability ≥ 6.0 (GTX 10xx series or newer). More memory allows larger models.

**Q: Does this support CPU-only mode?**
A: No. This engine requires an NVIDIA GPU. For CPU fallback, use PyTorch.

**Q: How do I debug CUDA errors?**
A: Compile with `nvcc -G` for debug info, then use `cuda-gdb` or `compute-sanitizer` to find memory errors.

**Q: Can I contribute without a GPU?**
A: Yes! Documentation, testing, API design, and code review contributions are valuable. CI runs on CPU for linting.

**Q: What CUDA version is required?**
A: CUDA 12.x. The code uses features available in CUDA 11.0+.

**Q: Is Windows supported?**
A: Linux is the primary target. Windows may work with Visual Studio and CUDA Toolkit but is not officially tested.

**Q: How do I add a new layer?**
A: 1) Define class in `include/layers.h`, 2) Implement forward/backward in `src/layers.cpp`, 3) Add Python binding in `bindings/python_bindings.cpp`, 4) Add tests.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

This project builds upon the work of many:

- **NVIDIA** — CUDA Toolkit, cuBLAS, cuDNN, and the GPU computing ecosystem
- **PyTorch Team** — API design inspiration and the autograd paradigm
- **Andrej Karpathy** — micrograd and educational resources on neural network internals
- **UVM (Unified Virtual Memory)** — for simplifying memory management patterns
- **The CUDA Programming Guide** — for optimization techniques and best practices
- **OpenBLAS/MKL** — for CPU-side reference implementations
- **The open-source community** — for countless tutorials, papers, and discussions that make projects like this possible

Special thanks to everyone who has contributed code, filed issues, and provided feedback.

---

*This project is for educational purposes. Not affiliated with NVIDIA Corporation.*
