// CUDA Neural Network Engine — CUDA Kernel Declarations
#pragma once

#include "tensor.h"

namespace cnn {
namespace kernels {

// Matrix operations (cuBLAS)
void matmul(const float* A, const float* B, float* C,
            int M, int N, int K, cublasHandle_t handle);

// Element-wise operations
void add_bias(float* output, const float* bias, int batch, int channels, int spatial);
void relu_forward(float* output, const float* input, int n);
void relu_backward(float* grad_input, const float* grad_output, const float* input, int n);
void sigmoid_forward(float* output, const float* input, int n);
void tanh_forward(float* output, const float* input, int n);
void leaky_relu_forward(float* output, const float* input, int n, float negative_slope);
void gelu_forward(float* output, const float* input, int n);
void silu_forward(float* output, const float* input, int n);

// Reduction
float sum(const float* data, int n);
void softmax(float* output, const float* input, int batch, int features);

// Loss
float cross_entropy_loss(const float* pred, const float* target, int batch, int features);
void cross_entropy_backward(float* grad_loss, const float* pred, const float* target, int batch, int features);

// Convolution
void conv2d_forward(const float* input, const float* weight, const float* bias,
                    float* output,
                    int batch, int in_channels, int out_channels,
                    int in_h, int in_w, int kernel_h, int kernel_w,
                    int stride_h, int stride_w, int pad_h, int pad_w);

// Pooling
void maxpool2d_forward(const float* input, float* output, int* indices,
                       int batch, int channels, int in_h, int in_w,
                       int pool_h, int pool_w, int stride_h, int stride_w);

// Utility
void fill_with_constant(float* data, float value, int n);
void copy_data(float* dst, const float* src, int n);
void scale_data(float* data, float factor, int n);

} // namespace kernels
} // namespace cnn
