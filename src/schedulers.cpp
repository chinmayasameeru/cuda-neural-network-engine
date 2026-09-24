// CUDA Neural Network Engine — Schedulers and Augmentations
#include "schedulers.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace cnn {

// ============================================================
// Learning Rate Schedulers
// ============================================================

StepLR::StepLR(float initial_lr, int step_size, float gamma)
    : initial_lr_(initial_lr), step_size_(step_size), gamma_(gamma) {}

float StepLR::get_lr(int step) const {
    int num_decays = step / step_size_;
    return initial_lr_ * std::pow(gamma_, num_decays);
}

CosineAnnealingLR::CosineAnnealingLR(float initial_lr, int T_max, float eta_min)
    : initial_lr_(initial_lr), T_max_(T_max), eta_min_(eta_min) {}

float CosineAnnealingLR::get_lr(int step) const {
    if (step >= T_max_) return eta_min_;
    return eta_min_ + 0.5f * (initial_lr_ - eta_min_) * (1.0f + std::cos(M_PI * step / T_max_));
}

WarmupCosineLR::WarmupCosineLR(float initial_lr, int warmup_steps, int total_steps)
    : initial_lr_(initial_lr), warmup_steps_(warmup_steps), total_steps_(total_steps) {}

float WarmupCosineLR::get_lr(int step) const {
    if (step < warmup_steps_) {
        return initial_lr_ * static_cast<float>(step) / warmup_steps_;
    }
    int progress = step - warmup_steps_;
    int max_progress = total_steps_ - warmup_steps_;
    return initial_lr_ * 0.5f * (1.0f + std::cos(M_PI * progress / max_progress));
}

ReduceLROnPlateau::ReduceLROnPlateau(float initial_lr, float factor, int patience, float min_lr)
    : current_lr_(initial_lr), factor_(factor), patience_(patience), min_lr_(min_lr),
      bad_epochs_(0), best_loss_(std::numeric_limits<float>::max()) {}

float ReduceLROnPlateau::get_lr(int) {
    return current_lr_;
}

void ReduceLROnPlateau::update_loss(float loss) {
    if (loss < best_loss_) {
        best_loss_ = loss;
        bad_epochs_ = 0;
    } else {
        bad_epochs_++;
        if (bad_epochs_ >= patience_) {
            current_lr_ = std::max(current_lr_ * factor_, min_lr_);
            bad_epochs_ = 0;
        }
    }
}

// ============================================================
// Data Augmentation Utilities
// ============================================================

std::vector<float> random_horizontal_flip(const std::vector<float>& image, int channels, int height, int width) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto result = image;
    if (dist(gen) < 0.5f) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width / 2; ++w) {
                    int idx1 = (c * height + h) * width + w;
                    int idx2 = (c * height + h) * width + (width - 1 - w);
                    std::swap(result[idx1], result[idx2]);
                }
            }
        }
    }
    return result;
}

std::vector<float> zero_pad(const std::vector<float>& image, int channels, int height, int width, int pad) {
    int new_h = height + 2 * pad;
    int new_w = width + 2 * pad;
    std::vector<float> result(channels * new_h * new_w, 0.0f);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx = (c * height + h) * width + w;
                int dst_idx = (c * new_h + h + pad) * new_w + (w + pad);
                result[dst_idx] = image[src_idx];
            }
        }
    }
    return result;
}

std::vector<float> random_crop(const std::vector<float>& image, int channels, int height, int width, int crop_h, int crop_w, int pad) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Pad first
    auto padded = zero_pad(image, channels, height, width, pad);
    int padded_h = height + 2 * pad;
    int padded_w = width + 2 * pad;

    std::uniform_int_distribution<int> h_dist(0, padded_h - crop_h);
    std::uniform_int_distribution<int> w_dist(0, padded_w - crop_w);

    int h_start = h_dist(gen);
    int w_start = w_dist(gen);

    std::vector<float> result(channels * crop_h * crop_w);
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_h; ++h) {
            for (int w = 0; w < crop_w; ++w) {
                int src_idx = (c * padded_h + h + h_start) * padded_w + (w + w_start);
                int dst_idx = (c * crop_h + h) * crop_w + w;
                result[dst_idx] = padded[src_idx];
            }
        }
    }
    return result;
}

std::vector<float> normalize(const std::vector<float>& image, const std::vector<float>& mean, const std::vector<float>& std) {
    auto result = image;
    int channels = mean.size();
    int pixels = image.size() / channels;

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < pixels; ++i) {
            int idx = c * pixels + i;
            result[idx] = (image[idx] - mean[c]) / std[c];
        }
    }
    return result;
}

} // namespace cnn
