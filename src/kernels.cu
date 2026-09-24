// CUDA Neural Network Engine — CUDA Kernel Implementations
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace cnn {
namespace kernels {

// ============================================================
// BLAS Operations (cuBLAS)
// ============================================================

void matmul(const float* A, const float* B, float* C,
            int M, int N, int K, cublasHandle_t handle)
{
    // C = A * B where A: M×K, B: K×N, C: M×N
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
}

// ============================================================
// Element-wise Kernels
// ============================================================

__global__ void add_bias_kernel(float* output, const float* bias,
                                 int batch, int channels, int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * spatial;
    if (idx >= total) return;
    int c = (idx / spatial) % channels;
    output[idx] += bias[c];
}

void add_bias(float* output, const float* bias, int batch, int channels, int spatial)
{
    int total = batch * channels * spatial;
    add_bias_kernel<<<(total + 255) / 256, 256>>>(output, bias, batch, channels, spatial);
}

// ReLU
__global__ void relu_forward_kernel(float* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = fmaxf(0.0f, input[idx]);
}

void relu_forward(float* output, const float* input, int n)
{
    relu_forward_kernel<<<(n + 255) / 256, 256>>>(output, input, n);
}

__global__ void relu_backward_kernel(float* grad_input, const float* grad_output,
                                     const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
}

void relu_backward(float* grad_input, const float* grad_output,
                   const float* input, int n)
{
    relu_backward_kernel<<<(n + 255) / 256, 256>>>(grad_input, grad_output, input, n);
}

// Sigmoid
__global__ void sigmoid_forward_kernel(float* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Clamp to avoid overflow
        x = fmaxf(-50.0f, fminf(50.0f, x));
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

void sigmoid_forward(float* output, const float* input, int n)
{
    sigmoid_forward_kernel<<<(n + 255) / 256, 256>>>(output, input, n);
}

// Tanh
__global__ void tanh_forward_kernel(float* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = tanhf(input[idx]);
}

void tanh_forward(float* output, const float* input, int n)
{
    tanh_forward_kernel<<<(n + 255) / 256, 256>>>(output, input, n);
}

// Leaky ReLU
__global__ void leaky_relu_forward_kernel(float* output, const float* input,
                                          int n, float negative_slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : negative_slope * x;
    }
}

void leaky_relu_forward(float* output, const float* input, int n, float negative_slope)
{
    leaky_relu_forward_kernel<<<(n + 255) / 256, 256>>>(output, input, n, negative_slope);
}

// GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void gelu_forward_kernel(float* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

void gelu_forward(float* output, const float* input, int n)
{
    gelu_forward_kernel<<<(n + 255) / 256, 256>>>(output, input, n);
}

// SiLU: x * sigmoid(x)
__global__ void silu_forward_kernel(float* output, const float* input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

void silu_forward(float* output, const float* input, int n)
{
    silu_forward_kernel<<<(n + 255) / 256, 256>>>(output, input, n);
}

// Softmax (numerically stable, per-row)
__global__ void softmax_kernel(float* output, const float* input,
                                int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    float max_val = in_row[0];
    for (int i = 1; i < cols; ++i)
        max_val = fmaxf(max_val, in_row[i]);

    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < cols; ++i)
        out_row[i] *= inv_sum;
}

void softmax(float* output, const float* input, int batch, int features)
{
    softmax_kernel<<<(batch + 255) / 256, 256>>>(output, input, batch, features);
}

// Sum reduction
__global__ void sum_kernel(const float* data, float* result, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (idx < n) val = data[idx];
    if (idx + blockDim.x < n) val += data[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

float sum(const float* data, int n)
{
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    int block = 256;
    int grid = (n + block * 2 - 1) / (block * 2);
    sum_kernel<<<grid, block, block * sizeof(float)>>>(data, d_result, n);

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return result;
}

// Cross Entropy Loss (softmax + negative log-likelihood)
__global__ void cross_entropy_kernel(const float* pred, const float* target,
                                      float* loss_out, int batch, int features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    const float* pred_row = pred + idx * features;
    const float* target_row = target + idx * features;

    float loss = 0.0f;
    for (int i = 0; i < features; ++i) {
        if (target_row[i] > 0.5f) {
            loss -= logf(fmaxf(pred_row[i], 1e-7f));
        }
    }
    atomicAdd(loss_out, loss / batch);
}

float cross_entropy_loss(const float* pred, const float* target, int batch, int features)
{
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    cross_entropy_kernel<<<(batch + 255) / 256, 256>>>(pred, target, d_loss, batch, features);

    float loss;
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    return loss;
}

// ============================================================
// 2D Convolution
// ============================================================

__global__ void conv2d_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_channels, int out_channels,
    int in_h, int in_w, int k_size, int stride, int pad)
{
    int out_h = (in_h + 2 * pad - k_size) / stride + 1;
    int out_w = (in_w + 2 * pad - k_size) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;

    // Decode 4D index
    int tmp = idx;
    int w = tmp % out_w; tmp /= out_w;
    int h = tmp % out_h; tmp /= out_h;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b = tmp;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < k_size; ++kh) {
            for (int kw = 0; kw < k_size; ++kw) {
                int ih = h * stride + kh - pad;
                int iw = w * stride + kw - pad;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                    int w_idx = ((oc * in_channels + ic) * k_size + kh) * k_size + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    output[idx] = sum;
}

void conv2d_forward(const float* input, const float* weight, const float* bias,
                    float* output,
                    int batch, int in_channels, int out_channels,
                    int in_h, int in_w, int k_h, int k_w,
                    int stride_h, int stride_w, int pad_h, int pad_w)
{
    // Simplified: use square kernels for now
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    int total = batch * out_channels * out_h * out_w;

    conv2d_forward_kernel<<<(total + 255) / 256, 256>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels,
        in_h, in_w, k_h, stride_h, pad_h);
}

// ============================================================
// Max Pooling
// ============================================================

__global__ void maxpool2d_forward_kernel(
    const float* input, float* output, int* indices,
    int batch, int channels, int in_h, int in_w,
    int pool_h, int pool_w, int stride_h, int stride_w)
{
    int out_h = (in_h - pool_h) / stride_h + 1;
    int out_w = (in_w - pool_w) / stride_w + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int tmp = idx;
    int w = tmp % out_w; tmp /= out_w;
    int h = tmp % out_h; tmp /= out_h;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;

    const float* in_base = input + (b * channels + c) * in_h * in_w;
    int h_start = h * stride_h;
    int w_start = w * stride_w;

    float max_val = -1e20f;
    int max_idx = 0;
    for (int ph = 0; ph < pool_h; ++ph) {
        for (int pw = 0; pw < pool_w; ++pw) {
            int cur_idx = (h_start + ph) * in_w + (w_start + pw);
            float val = in_base[cur_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = cur_idx;
            }
        }
    }
    output[idx] = max_val;
    indices[idx] = max_idx;
}

void maxpool2d_forward(const float* input, float* output, int* indices,
                       int batch, int channels, int in_h, int in_w,
                       int pool_h, int pool_w, int stride_h, int stride_w)
{
    int out_h = (in_h - pool_h) / stride_h + 1;
    int out_w = (in_w - pool_w) / stride_w + 1;
    int total = batch * channels * out_h * out_w;

    maxpool2d_forward_kernel<<<(total + 255) / 256, 256>>>(
        input, output, indices,
        batch, channels, in_h, in_w,
        pool_h, pool_w, stride_h, stride_w);
}

// ============================================================
// Utility Kernels
// ============================================================

__global__ void fill_kernel(float* data, float value, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

void fill_with_constant(float* data, float value, int n)
{
    fill_kernel<<<(n + 255) / 256, 256>>>(data, value, n);
}

__global__ void copy_kernel(float* dst, const float* src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

void copy_data(float* dst, const float* src, int n)
{
    copy_kernel<<<(n + 255) / 256, 256>>>(dst, src, n);
}

__global__ void scale_kernel(float* data, float factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= factor;
}

void scale_data(float* data, float factor, int n)
{
    scale_kernel<<<(n + 255) / 256, 256>>>(data, factor, n);
}

} // namespace kernels
} // namespace cnn
