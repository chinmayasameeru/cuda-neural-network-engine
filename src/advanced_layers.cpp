// CUDA Neural Network Engine — Advanced Layer Implementations
#include "advanced_layers.h"
#include "kernels.cuh"
#include <random>
#include <cmath>

namespace cnn {

// ============================================================
// Residual Block
// ============================================================
ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride)
    : in_c_(in_channels), out_c_(out_channels), stride_(stride)
{
    conv1_ = std::make_shared<Conv2D>(in_channels, out_channels, 3, stride, 1, false);
    bn1_ = std::make_shared<BatchNorm2D>(out_channels);
    relu1_ = std::make_shared<ReLU>();
    conv2_ = std::make_shared<Conv2D>(out_channels, out_channels, 3, 1, 1, false);
    bn2_ = std::make_shared<BatchNorm2D>(out_channels);
    relu2_ = std::make_shared<ReLU>();

    use_shortcut_ = (stride != 1 || in_channels != out_channels);
    if (use_shortcut_) {
        shortcut_ = std::make_shared<Conv2D>(in_channels, out_channels, 1, stride, 0, false);
    }
}

TensorPtr ResidualBlock::forward(TensorPtr input) {
    // Main path
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = relu1_->forward(x);
    x = conv2_->forward(x);
    x = bn2_->forward(x);

    // Shortcut
    auto shortcut = use_shortcut_ ? shortcut_->forward(input) : input;

    // Add (element-wise addition via CPU for simplicity)
    auto x_host = x->to_cpu();
    auto shortcut_host = shortcut->to_cpu();
    for (size_t i = 0; i < x_host.size(); ++i) {
        x_host[i] += shortcut_host[i];
    }
    x->from_cpu(x_host);

    return relu2_->forward(x);
}

std::vector<TensorPtr> ResidualBlock::parameters() {
    std::vector<TensorPtr> params;
    auto p1 = conv1_->parameters();
    auto p2 = bn1_->parameters();
    auto p3 = conv2_->parameters();
    auto p4 = bn2_->parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());
    if (use_shortcut_) {
        auto ps = shortcut_->parameters();
        params.insert(params.end(), ps.begin(), ps.end());
    }
    return params;
}

void ResidualBlock::zero_grad() {
    conv1_->zero_grad();
    bn1_->zero_grad();
    conv2_->zero_grad();
    bn2_->zero_grad();
    if (use_shortcut_) shortcut_->zero_grad();
}

// ============================================================
// LSTM Cell
// ============================================================
LSTMCell::LSTMCell(int input_size, int hidden_size)
    : in_size_(input_size), hidden_size_(hidden_size)
{
    // Xavier initialization
    float limit = std::sqrt(6.0f / (input_size + hidden_size));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Forget gate weights
    Wf_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wf_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wf_data) v = dist(gen);
    Wf_->from_cpu(wf_data);
    bf_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> bf_data(hidden_size, 0.0f);
    bf_->from_cpu(bf_data);

    // Input gate weights
    Wi_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wi_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wi_data) v = dist(gen);
    Wi_->from_cpu(wi_data);
    bi_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> bi_data(hidden_size, 0.0f);
    bi_->from_cpu(bi_data);

    // Cell gate weights
    Wc_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wc_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wc_data) v = dist(gen);
    Wc_->from_cpu(wc_data);
    bc_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> bc_data(hidden_size, 0.0f);
    bc_->from_cpu(bc_data);

    // Output gate weights
    Wo_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wo_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wo_data) v = dist(gen);
    Wo_->from_cpu(wo_data);
    bo_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> bo_data(hidden_size, 0.0f);
    bo_->from_cpu(bo_data);
}

TensorPtr LSTMCell::forward(TensorPtr input) {
    // input shape: [batch, in_size]
    int batch = input->shape[0];

    // Initialize states on first call
    if (first_call_) {
        h_prev_ = std::make_shared<Tensor>(std::vector<int>{batch, hidden_size_}, false);
        c_prev_ = std::make_shared<Tensor>(std::vector<int>{batch, hidden_size_}, false);
        first_call_ = false;
    }

    // Concatenate input and previous hidden state: [x, h_prev]
    auto input_host = input->to_cpu();
    auto h_prev_host = h_prev_->to_cpu();
    std::vector<float> combined;
    combined.reserve(input_host.size() + h_prev_host.size());
    combined.insert(combined.end(), input_host.begin(), input_host.end());
    combined.insert(combined.end(), h_prev_host.begin(), h_prev_host.end());
    TensorPtr combined_t = std::make_shared<Tensor>(std::vector<int>{batch, in_size_ + hidden_size_}, false);
    combined_t->from_cpu(combined);

    // Gates
    // Forget gate: f = sigmoid(Wf * [x, h] + bf)
    // Input gate: i = sigmoid(Wi * [x, h] + bi)
    // Cell gate: c = tanh(Wc * [x, h] + bc)
    // Output gate: o = sigmoid(Wo * [x, h] + bo)

    // For simplicity, compute gates via CPU (full GPU implementation would need batched ops)
    auto c_prev_host = c_prev_->to_cpu();
    std::vector<float> h_new(hidden_size_ * batch, 0.0f);
    std::vector<float> c_new(hidden_size_ * batch, 0.0f);

    // Simplified LSTM forward (CPU fallback)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < hidden_size_; ++h) {
            float f = 0, i = 0, c = 0, o = 0;
            for (int j = 0; j < in_size_ + hidden_size_; ++j) {
                float xj = (j < in_size_) ? input_host[b * in_size_ + j] : h_prev_host[b * hidden_size_ + (j - in_size_)];
                f += Wf_->data[h * (in_size_ + hidden_size_) + j] * xj;
                i += Wi_->data[h * (in_size_ + hidden_size_) + j] * xj;
                c += Wc_->data[h * (in_size_ + hidden_size_) + j] * xj;
                o += Wo_->data[h * (in_size_ + hidden_size_) + j] * xj;
            }
            f = 1.0f / (1.0f + std::exp(-f - bf_->data[h]));
            i = 1.0f / (1.0f + std::exp(-i - bi_->data[h]));
            c = std::tanh(c + bc_->data[h]);
            o = 1.0f / (1.0f + std::exp(-o - bo_->data[h]));

            c_new[b * hidden_size_ + h] = f * c_prev_host[b * hidden_size_ + h] + i * c;
            h_new[b * hidden_size_ + h] = o * std::tanh(c_new[b * hidden_size_ + h]);
        }
    }

    // Update state
    c_prev_->from_cpu(c_new);
    h_prev_->from_cpu(h_new);

    auto output = std::make_shared<Tensor>(std::vector<int>{batch, hidden_size_}, false);
    output->from_cpu(h_new);
    return output;
}

std::vector<TensorPtr> LSTMCell::parameters() {
    return {Wf_, bf_, Wi_, bi_, Wc_, bc_, Wo_, bo_};
}

void LSTMCell::zero_grad() {
    Wf_->zero_grad(); bf_->zero_grad();
    Wi_->zero_grad(); bi_->zero_grad();
    Wc_->zero_grad(); bc_->zero_grad();
    Wo_->zero_grad(); bo_->zero_grad();
}

void LSTMCell::reset_state() {
    first_call_ = true;
}

// ============================================================
// GRU Cell
// ============================================================
GRUCell::GRUCell(int input_size, int hidden_size)
    : in_size_(input_size), hidden_size_(hidden_size)
{
    float limit = std::sqrt(6.0f / (input_size + hidden_size));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Update gate
    Wz_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wz_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wz_data) v = dist(gen);
    Wz_->from_cpu(wz_data);
    bz_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> bz_data(hidden_size, 0.0f);
    bz_->from_cpu(bz_data);

    // Reset gate
    Wr_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wr_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wr_data) v = dist(gen);
    Wr_->from_cpu(wr_data);
    br_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> br_data(hidden_size, 0.0f);
    br_->from_cpu(br_data);

    // Hidden gate
    Wh_ = std::make_shared<Tensor>(std::vector<int>{hidden_size, input_size + hidden_size}, true);
    std::vector<float> wh_data(hidden_size * (input_size + hidden_size));
    for (auto& v : wh_data) v = dist(gen);
    Wh_->from_cpu(wh_data);
    bh_ = std::make_shared<Tensor>(std::vector<int>{hidden_size}, true);
    std::vector<float> bh_data(hidden_size, 0.0f);
    bh_->from_cpu(bh_data);
}

TensorPtr GRUCell::forward(TensorPtr input) {
    int batch = input->shape[0];

    if (first_call_) {
        h_prev_ = std::make_shared<Tensor>(std::vector<int>{batch, hidden_size_}, false);
        first_call_ = false;
    }

    auto input_host = input->to_cpu();
    auto h_prev_host = h_prev_->to_cpu();
    std::vector<float> h_new(hidden_size_ * batch, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < hidden_size_; ++h) {
            float z = 0, r = 0, hh = 0;
            for (int j = 0; j < in_size_ + hidden_size_; ++j) {
                float xj = (j < in_size_) ? input_host[b * in_size_ + j] : h_prev_host[b * hidden_size_ + (j - in_size_)];
                z += Wz_->data[h * (in_size_ + hidden_size_) + j] * xj;
                r += Wr_->data[h * (in_size_ + hidden_size_) + j] * xj;
                hh += Wh_->data[h * (in_size_ + hidden_size_) + j] * xj;
            }
            z = 1.0f / (1.0f + std::exp(-z - bz_->data[h]));
            r = 1.0f / (1.0f + std::exp(-r - br_->data[h]));
            float h_hat = std::tanh(r * h_prev_host[b * hidden_size_ + h] + bh_->data[h]);
            h_new[b * hidden_size_ + h] = (1.0f - z) * h_prev_host[b * hidden_size_ + h] + z * h_hat;
        }
    }

    h_prev_->from_cpu(h_new);
    auto output = std::make_shared<Tensor>(std::vector<int>{batch, hidden_size_}, false);
    output->from_cpu(h_new);
    return output;
}

std::vector<TensorPtr> GRUCell::parameters() {
    return {Wz_, bz_, Wr_, br_, Wh_, bh_};
}

void GRUCell::zero_grad() {
    Wz_->zero_grad(); bz_->zero_grad();
    Wr_->zero_grad(); br_->zero_grad();
    Wh_->zero_grad(); bh_->zero_grad();
}

void GRUCell::reset_state() {
    first_call_ = true;
}

// ============================================================
// Layer Normalization
// ============================================================
LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps)
{
    gamma_ = std::make_shared<Tensor>(std::vector<int>{normalized_shape}, true);
    beta_ = std::make_shared<Tensor>(std::vector<int>{normalized_shape}, true);
    std::vector<float> ones(normalized_shape, 1.0f);
    std::vector<float> zeros(normalized_shape, 0.0f);
    gamma_->from_cpu(ones);
    beta_->from_cpu(zeros);
}

TensorPtr LayerNorm::forward(TensorPtr input) {
    auto input_host = input->to_cpu();
    auto output_host = input_host;

    // Normalize along last dimension
    int batch = input->numel / normalized_shape_;
    for (int b = 0; b < batch; ++b) {
        float sum = 0.0f;
        for (int i = 0; i < normalized_shape_; ++i) {
            sum += input_host[b * normalized_shape_ + i];
        }
        float mean = sum / normalized_shape_;

        float var_sum = 0.0f;
        for (int i = 0; i < normalized_shape_; ++i) {
            float diff = input_host[b * normalized_shape_ + i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / normalized_shape_;
        float inv_std = 1.0f / std::sqrt(var + eps_);

        for (int i = 0; i < normalized_shape_; ++i) {
            float normalized = (input_host[b * normalized_shape_ + i] - mean) * inv_std;
            output_host[b * normalized_shape_ + i] = gamma_->data[i] * normalized + beta_->data[i];
        }
    }

    auto output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    output->from_cpu(output_host);
    return output;
}

std::vector<TensorPtr> LayerNorm::parameters() {
    return {gamma_, beta_};
}

} // namespace cnn
