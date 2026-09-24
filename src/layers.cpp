// CUDA Neural Network Engine — Layer Implementations
#include "layers.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <random>
#include <cmath>

namespace cnn {

// ============================================================
// Linear Layer
// ============================================================

Linear::Linear(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias)
{
    float std_dev = std::sqrt(2.0f / in_features);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, std_dev);

    weight_ = std::make_shared<Tensor>(std::vector<int>{out_features, in_features}, true);
    std::vector<float> w_data(out_features * in_features);
    for (auto& v : w_data) v = dist(gen);
    weight_->from_cpu(w_data);

    if (use_bias_) {
        bias_ = std::make_shared<Tensor>(std::vector<int>{out_features}, true);
        std::vector<float> b_data(out_features, 0.0f);
        bias_->from_cpu(b_data);
    }
}

TensorPtr Linear::forward(TensorPtr input) {
    int batch = input->shape[0];
    auto output = std::make_shared<Tensor>(std::vector<int>{batch, out_features_}, true);

    kernels::matmul(input->data, weight_->data, output->data,
                    batch, out_features_, in_features_, blas_handle());

    if (use_bias_) {
        kernels::add_bias(output->data, bias_->data, batch, out_features_, 1);
    }

    if (output->requires_grad) {
        output->prev = {input, weight_};
        if (use_bias_) output->prev.push_back(bias_);
        output->backward_fn = [this, input, output]() {
            int batch = input->shape[0];

            // grad_weight = grad_output^T @ input
            if (weight_->requires_grad) {
                if (!weight_->grad) {
                    weight_->grad = static_cast<float*>(
                        malloc(weight_->numel * sizeof(float)));
                    memset(weight_->grad, 0, weight_->numel * sizeof(float));
                }
                const float alpha = 1.0f, beta = 1.0f;
                cublasSgemm(blas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                            in_features_, out_features_, batch,
                            &alpha,
                            input->data, in_features_,
                            output->grad, out_features_,
                            &beta,
                            weight_->grad, in_features_);
            }

            // grad_input = grad_output @ weight
            if (input->requires_grad) {
                if (!input->grad) {
                    input->grad = static_cast<float*>(
                        malloc(input->numel * sizeof(float)));
                    memset(input->grad, 0, input->numel * sizeof(float));
                }
                const float alpha = 1.0f, beta = 1.0f;
                cublasSgemm(blas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                            in_features_, batch, out_features_,
                            &alpha,
                            weight_->data, in_features_,
                            output->grad, out_features_,
                            &beta,
                            input->grad, in_features_);
            }

            // grad_bias = sum(grad_output, dim=0)
            if (use_bias_ && bias_->requires_grad) {
                if (!bias_->grad) {
                    bias_->grad = static_cast<float*>(
                        malloc(bias_->numel * sizeof(float)));
                    memset(bias_->grad, 0, bias_->numel * sizeof(float));
                }
                float* ones;
                cudaMalloc(&ones, batch * sizeof(float));
                kernels::fill_with_constant(ones, 1.0f, batch);
                const float alpha = 1.0f, beta = 1.0f;
                cublasSgemv(blas_handle(), CUBLAS_OP_T,
                            batch, out_features_,
                            &alpha,
                            output->grad, batch,
                            ones, 1,
                            &beta,
                            bias_->grad, 1);
                cudaFree(ones);
            }
        };
    }

    return output;
}

// ============================================================
// Conv2D Layer
// ============================================================

Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, bool use_bias)
    : in_c_(in_channels), out_c_(out_channels),
      k_size_(kernel_size), stride_(stride), pad_(padding), use_bias_(use_bias)
{
    float std_dev = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, std_dev);

    weight_ = std::make_shared<Tensor>(std::vector<int>{out_c_, in_c_, k_size_, k_size_}, true);
    int w_size = out_c_ * in_c_ * k_size_ * k_size_;
    std::vector<float> w_data(w_size);
    for (auto& v : w_data) v = dist(gen);
    weight_->from_cpu(w_data);

    if (use_bias_) {
        bias_ = std::make_shared<Tensor>(std::vector<int>{out_c_}, true);
        std::vector<float> b_data(out_c_, 0.0f);
        bias_->from_cpu(b_data);
    }
}

TensorPtr Conv2D::forward(TensorPtr input) {
    int batch = input->shape[0];
    int in_h = input->shape[2];
    int in_w = input->shape[3];
    int out_h = (in_h + 2 * pad_ - k_size_) / stride_ + 1;
    int out_w = (in_w + 2 * pad_ - k_size_) / stride_ + 1;

    auto output = std::make_shared<Tensor>(
        std::vector<int>{batch, out_c_, out_h, out_w}, true);

    kernels::conv2d_forward(
        input->data, weight_->data, bias_ ? bias_->data : nullptr,
        output->data,
        batch, in_c_, out_c_,
        in_h, in_w, k_size_, k_size_,
        stride_, stride_, pad_, pad_);

    return output;
}

// ============================================================
// BatchNorm2D Layer
// ============================================================

BatchNorm2D::BatchNorm2D(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum)
{
    gamma_ = std::make_shared<Tensor>(std::vector<int>{num_features}, true);
    beta_ = std::make_shared<Tensor>(std::vector<int>{num_features}, true);
    running_mean_ = std::make_shared<Tensor>(std::vector<int>{num_features}, false);
    running_var_ = std::make_shared<Tensor>(std::vector<int>{num_features}, false);

    std::vector<float> ones(num_features, 1.0f);
    std::vector<float> zeros(num_features, 0.0f);
    gamma_->from_cpu(ones);
    beta_->from_cpu(zeros);
}

TensorPtr BatchNorm2D::forward(TensorPtr input) {
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int spatial = H * W;

    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    auto input_host = input->to_cpu();
    auto output_host = output->to_cpu();

    for (int c = 0; c < C; ++c) {
        // Compute mean
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                sum += input_host[n * C * spatial + c * spatial + s];
            }
        }
        float mean = sum / (N * spatial);

        // Compute variance
        float var_sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                float diff = input_host[n * C * spatial + c * spatial + s] - mean;
                var_sum += diff * diff;
            }
        }
        float var = var_sum / (N * spatial);

        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps_);
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < spatial; ++s) {
                int idx = n * C * spatial + c * spatial + s;
                float normalized = (input_host[idx] - mean) * inv_std;
                output_host[idx] = gamma_->data[c] * normalized + beta_->data[c];
            }
        }

        // Update running stats
        if (training_) {
            running_mean_->data[c] = momentum_ * running_mean_->data[c] + (1.0f - momentum_) * mean;
            running_var_->data[c] = momentum_ * running_var_->data[c] + (1.0f - momentum_) * var;
        }
    }

    output->from_cpu(output_host);
    return output;
}

// ============================================================
// MaxPool2D Layer
// ============================================================

MaxPool2D::MaxPool2D(int pool_size, int stride)
    : pool_size_(pool_size), stride_(stride > 0 ? stride : pool_size) {}

TensorPtr MaxPool2D::forward(TensorPtr input) {
    int batch = input->shape[0];
    int channels = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];
    int out_h = (in_h - pool_size_) / stride_ + 1;
    int out_w = (in_w - pool_size_) / stride_ + 1;

    auto output = std::make_shared<Tensor>(
        std::vector<int>{batch, channels, out_h, out_w}, input->requires_grad);

    int* indices;
    cudaMalloc(&indices, batch * channels * out_h * out_w * sizeof(int));

    kernels::maxpool2d_forward(
        input->data, output->data, indices,
        batch, channels, in_h, in_w,
        pool_size_, pool_size_, stride_, stride_);

    cudaFree(indices);
    return output;
}

// ============================================================
// AdaptiveAvgPool2D Layer
// ============================================================

AdaptiveAvgPool2D::AdaptiveAvgPool2D(int output_h, int output_w)
    : output_h_(output_h), output_w_(output_w) {}

TensorPtr AdaptiveAvgPool2D::forward(TensorPtr input) {
    int N = input->shape[0];
    int C = input->shape[1];
    int in_h = input->shape[2];
    int in_w = input->shape[3];

    auto output = std::make_shared<Tensor>(
        std::vector<int>{N, C, output_h_, output_w_}, input->requires_grad);

    auto input_host = input->to_cpu();
    auto output_host = output->to_cpu();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < output_h_; ++oh) {
                for (int ow = 0; ow < output_w_; ++ow) {
                    int h_start = (oh * in_h) / output_h_;
                    int h_end = ((oh + 1) * in_h + output_h_ - 1) / output_h_;
                    int w_start = (ow * in_w) / output_w_;
                    int w_end = ((ow + 1) * in_w + output_w_ - 1) / output_w_;

                    float sum = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            sum += input_host[((n * C + c) * in_h + h) * in_w + w];
                            count++;
                        }
                    }
                    output_host[((n * C + c) * output_h_ + oh) * output_w_ + ow] = sum / count;
                }
            }
        }
    }

    output->from_cpu(output_host);
    return output;
}

// ============================================================
// Activation Layers
// ============================================================

TensorPtr ReLU::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::relu_forward(output->data, input->data, input->numel);

    if (output->requires_grad) {
        output->prev = {input};
        output->backward_fn = [input, output]() {
            if (input->requires_grad) {
                if (!input->grad) {
                    input->grad = static_cast<float*>(
                        malloc(input->numel * sizeof(float)));
                    memset(input->grad, 0, input->numel * sizeof(float));
                }
                kernels::relu_backward(input->grad, output->grad, input->data, input->numel);
            }
        };
    }
    return output;
}

TensorPtr Sigmoid::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::sigmoid_forward(output->data, input->data, input->numel);
    return output;
}

TensorPtr Tanh::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::tanh_forward(output->data, input->data, input->numel);
    return output;
}

TensorPtr LeakyReLU::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::leaky_relu_forward(output->data, input->data, input->numel, negative_slope_);
    return output;
}

TensorPtr GELU::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::gelu_forward(output->data, input->data, input->numel);
    return output;
}

TensorPtr SiLU::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::silu_forward(output->data, input->data, input->numel);
    return output;
}

TensorPtr Softmax::forward(TensorPtr input) {
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    int batch = input->shape[0];
    int features = input->numel / batch;
    kernels::softmax(output->data, input->data, batch, features);
    return output;
}

// ============================================================
// Flatten Layer
// ============================================================

TensorPtr Flatten::forward(TensorPtr input) {
    int batch = input->shape[0];
    int rest = input->numel / batch;
    auto output = std::make_shared<Tensor>(std::vector<int>{batch, rest}, input->requires_grad);
    kernels::copy_data(output->data, input->data, input->numel);
    return output;
}

// ============================================================
// Dropout Layer
// ============================================================

TensorPtr Dropout::forward(TensorPtr input) {
    if (!training_) return input;
    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    kernels::copy_data(output->data, input->data, input->numel);
    kernels::scale_data(output->data, 1.0f / (1.0f - p_), input->numel);
    return output;
}

} // namespace cnn
