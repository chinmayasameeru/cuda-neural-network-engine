// CUDA Neural Network Engine — Schedulers and Augmentations (v0.2.0)
#pragma once

#include <vector>
#include <functional>

namespace cnn {

// ============================================================
// Learning Rate Schedulers
// ============================================================
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step) const = 0;
    virtual std::string name() const = 0;
};

// Step LR: decay by gamma every step_size steps
class StepLR : public LRScheduler {
public:
    StepLR(float initial_lr, int step_size, float gamma = 0.1f);

    float get_lr(int step) const override;
    std::string name() const override { return "StepLR"; }

private:
    float initial_lr_;
    int step_size_;
    float gamma_;
};

// Cosine Annealing: cosine schedule from initial_lr to eta_min
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(float initial_lr, int T_max, float eta_min = 0.0f);

    float get_lr(int step) const override;
    std::string name() const override { return "CosineAnnealingLR"; }

private:
    float initial_lr_;
    int T_max_;
    float eta_min_;
};

// Linear Warmup + Cosine Decay
class WarmupCosineLR : public LRScheduler {
public:
    WarmupCosineLR(float initial_lr, int warmup_steps, int total_steps);

    float get_lr(int step) const override;
    std::string name() const override { return "WarmupCosineLR"; }

private:
    float initial_lr_;
    int warmup_steps_;
    int total_steps_;
};

// Reduce on Plateau: reduce LR when loss stops improving
class ReduceLROnPlateau : public LRScheduler {
public:
    ReduceLROnPlateau(float initial_lr, float factor = 0.1f, int patience = 10, float min_lr = 1e-7f);

    float get_lr(int step) override;  // non-const because it tracks state
    std::string name() const override { return "ReduceLROnPlateau"; }
    void update_loss(float loss);

private:
    float current_lr_;
    float factor_;
    int patience_;
    float min_lr_;
    int bad_epochs_;
    float best_loss_;
};

// ============================================================
// Data Augmentation Utilities
// ============================================================

// Random horizontal flip (for images)
std::vector<float> random_horizontal_flip(const std::vector<float>& image, int channels, int height, int width);

// Random crop with padding
std::vector<float> random_crop(const std::vector<float>& image, int channels, int height, int width, int crop_h, int crop_w, int pad);

// Normalize with mean and std
std::vector<float> normalize(const std::vector<float>& image, const std::vector<float>& mean, const std::vector<float>& std);

// Zero-pad an image
std::vector<float> zero_pad(const std::vector<float>& image, int channels, int height, int width, int pad);

} // namespace cnn
