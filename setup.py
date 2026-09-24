from setuptools import setup, Extension
import os

cuda_include = "/usr/local/cuda/include"
cuda_lib = "/usr/local/cuda/lib64"

setup(
    name="cnn_engine",
    version="0.1.0",
    description="CUDA Neural Network Engine",
    ext_modules=[
        Extension(
            "cnn_engine",
            sources=[
                "bindings/python_bindings.cpp",
                "src/tensor.cpp",
                "src/layers.cpp",
                "src/optimizers.cpp",
            ],
            include_dirs=["include", cuda_include, "/usr/include/python3.10"],
            library_dirs=[cuda_lib],
            libraries=["cudart", "cublas", "stdc++"],
            extra_compile_args=["-std=c++17", "-O3", "-fPIC"],
        )
    ],
)
