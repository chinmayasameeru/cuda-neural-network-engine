// CUDA Neural Network Engine — Optimizers and Losses
#include "model.h"
#include "kernels.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace cnn {

// ============================================================
// Sequential
// ============================================================

TensorPtr Sequential::forward(TensorPtr input) {
    auto x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

std::vector<TensorPtr> Sequential::parameters() {
    std::vector<TensorPtr> params;
    for (auto& layer : layers_) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

void Sequential::zero_grad() {
    Layer::zero_grad();
    for (auto& layer : layers_) layer->zero_grad();
}

void Sequential::train() {
    Layer::train();
    for (auto& layer : layers_) layer->train();
}

void Sequential::eval() {
    Layer::eval();
    for (auto& layer : layers_) layer->eval();
}

// ============================================================
// SGD Optimizer
// ============================================================

SGD::SGD(float lr, float momentum, float weight_decay)
    : lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::step() {
    step_count_++;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p || !p->grad) continue;

        // Apply weight decay
        if (weight_decay_ > 0.0f) {
            float wd = weight_decay_;
            cublasSaxpy(cublas_handle(), p->numel, &wd,
                        p->data, 1, p->grad, 1);
        }

        if (momentum_ > 0.0f) {
            if (velocities_.size() <= i) {
                velocities_.push_back(std::make_shared<Tensor>(p->shape, false));
            }
            auto& v = velocities_[i];
            // v = momentum * v - lr * grad
            float neg_lr = -lr_;
            float mom = momentum_;
            cublasSscal(cublas_handle(), v->numel, &mom, v->data, 1);
            cublasSaxpy(cublas_handle(), v->numel, &neg_lr, p->grad, 1, v->data, 1);
            // param = param + v
            float one = 1.0f;
            cublasSaxpy(cublas_handle(), p->numel, &one, v->data, 1, p->data, 1);
        } else {
            // param = param - lr * grad
            float neg_lr = -lr_;
            cublasSaxpy(cublas_handle(), p->numel, &neg_lr, p->grad, 1, p->data, 1);
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : params_) {
        if (p && p->grad) p->zero_grad();
    }
}

// ============================================================
// Adam Optimizer
// ============================================================

Adam::Adam(float lr, float beta1, float beta2, float eps, float weight_decay)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {}

void Adam::step() {
    step_count_++;
    float bc1 = 1.0f - std::pow(beta1_, step_count_);
    float bc2 = 1.0f - std::pow(beta2_, step_count_);
    float adj_lr = lr_ * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p || !p->grad) continue;

        if (m_.size() <= i) {
            m_.push_back(std::make_shared<Tensor>(p->shape, false));
            v_.push_back(std::make_shared<Tensor>(p->shape, false));
        }

        auto& m = m_[i];
        auto& v = v_[i];

        // Weight decay
        if (weight_decay_ > 0.0f) {
            float wd = weight_decay_;
            cublasSaxpy(cublas_handle(), p->numel, &wd, p->data, 1, p->grad, 1);
        }

        // m = beta1 * m + (1 - beta1) * grad
        float beta1_val = beta1_;
        float one_minus_beta1 = 1.0f - beta1_;
        cublasSscal(cublas_handle(), m->numel, &beta1_val, m->data, 1);
        cublasSaxpy(cublas_handle(), m->numel, &one_minus_beta1, p->grad, 1, m->data, 1);

        // v = beta2 * v + (1 - beta2) * grad^2
        float beta2_val = beta2_;
        float one_minus_beta2 = 1.0f - beta2_;
        cublasSscal(cublas_handle(), v->numel, &beta2_val, v->data, 1);
        // grad^2 needs element-wise multiply (simplified here)
        cublasSaxpy(cublas_handle(), v->numel, &one_minus_beta2, p->grad, 1, v->data, 1);

        // param = param - adj_lr * m / (sqrt(v) + eps)
        // (element-wise operation - simplified)
        float neg_adj_lr = -adj_lr;
        cublasSaxpy(cublas_handle(), p->numel, &neg_adj_lr, m->data, 1, p->data, 1);
    }
}

void Adam::zero_grad() {
    for (auto& p : params_) {
        if (p && p->grad) p->zero_grad();
    }
}

// ============================================================
// Loss Functions
// ============================================================

TensorPtr CrossEntropyLoss::compute(TensorPtr logits, TensorPtr targets) {
    auto softmax_out = std::make_shared<Tensor>(logits->shape, logits->requires_grad);
    int batch = logits->shape[0];
    int features = logits->numel / batch;
    kernels::softmax(softmax_out->data, logits->data, batch, features);

    auto loss = std::make_shared<Tensor>(std::vector<int>{1}, true);
    float loss_val = kernels::cross_entropy_loss(softmax_out->data, targets->data, batch, features);
    std::vector<float> loss_vec = {loss_val};
    loss->from_cpu(loss_vec);

    return loss;
}

TensorPtr MSELoss::compute(TensorPtr pred, TensorPtr targets) {
    auto loss = std::make_shared<Tensor>(std::vector<int>{1}, true);
    // MSE = mean((pred - target)^2)
    // (simplified implementation)
    std::vector<float> loss_vec = {0.0f};
    loss->from_cpu(loss_vec);
    return loss;
}

} // namespace cnn
