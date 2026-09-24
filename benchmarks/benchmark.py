#!/usr/bin/env python3
"""
CUDA Neural Network Engine — Benchmark Suite
"""
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import cnn_engine as cnn
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("cnn_engine not available — build with: python setup.py build_ext --inplace")

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


def benchmark_matmul():
    print("\n=== Matrix Multiplication ===")
    sizes = [(128, 256, 512), (256, 512, 1024)]

    for M, N, K in sizes:
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        # Warmup
        for _ in range(3):
            _ = a @ b

        start = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        cpu_time = (time.perf_counter() - start) / 10
        print(f"  CPU ({M}x{K} @ {K}x{N}): {cpu_time*1000:.2f}ms")

        if HAS_TORCH:
            a_t = torch.randn(M, K, device='cuda')
            b_t = torch.randn(K, N, device='cuda')
            for _ in range(3):
                _ = a_t @ b_t
            start = time.perf_counter()
            for _ in range(10):
                _ = a_t @ b_t
            gpu_time = (time.perf_counter() - start) / 10
            print(f"  GPU ({M}x{K} @ {K}x{N}): {gpu_time*1000:.2f}ms")
            print(f"  Speedup: {cpu_time/gpu_time:.1f}x")


def benchmark_training():
    print("\n=== Training Loop ===")
    if not HAS_TORCH:
        print("  SKIP — PyTorch not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    X = torch.randn(64, 784, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    elapsed = (time.perf_counter() - start) / 50
    print(f"  MLP training: {elapsed*1000:.2f}ms/iter on {device}")
    print(f"  Throughput: {64/elapsed:.0f} samples/sec")


def benchmark_cnn():
    print("\n=== CNN Inference ===")
    if not HAS_TORCH:
        print("  SKIP — PyTorch not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 8 * 8, 10)
    ).to(device)

    x = torch.randn(32, 3, 32, 32, device=device)

    for _ in range(5):
        _ = model(x)

    start = time.perf_counter()
    for _ in range(100):
        _ = model(x)

    elapsed = (time.perf_counter() - start) / 100
    print(f"  CNN inference: {elapsed*1000:.2f}ms/batch on {device}")
    print(f"  Throughput: {32/elapsed:.0f} images/sec")


def main():
    print("=" * 60)
    print("  CUDA Neural Network Engine — Benchmarks")
    print("=" * 60)
    benchmark_matmul()
    benchmark_training()
    benchmark_cnn()
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
