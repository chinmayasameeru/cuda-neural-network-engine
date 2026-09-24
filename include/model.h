// CUDA Neural Network Engine — Model, Optimizers, Losses
#pragma once

#include "layers.h"
#include <vector>
#include <memory>

namespace cnn {

class Sequential : public Layer {
public:
    Sequential() = default;

    template<typename T, typename... Args>
    void add(Args&&... args) {
        layers_.push_back(std::make_shared<T>(std::forward<Args>(args)...));
    }

    void add(LayerPtr layer) { if (layer) layers_.push_back(std::move(layer)); }

    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Sequential"; }
    std::vector<TensorPtr> parameters() override;
    void zero_grad() override;
    void train() override;
    void eval() override;

private:
    std::vector<LayerPtr> layers_;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    void add_param(TensorPtr param) { params_.push_back(std::move(param)); }

protected:
    std::vector<TensorPtr> params_;
    int step_count_ = 0;
};

class SGD : public Optimizer {
public:
    SGD(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;

private:
    float lr_, momentum_, weight_decay_;
    std::vector<TensorPtr> velocities_;
};

class Adam : public Optimizer {
public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<TensorPtr> m_;
    std::vector<TensorPtr> v_;
};

class CrossEntropyLoss {
public:
    TensorPtr compute(TensorPtr logits, TensorPtr targets);
};

class MSELoss {
public:
    TensorPtr compute(TensorPtr pred, TensorPtr targets);
};

} // namespace cnn
