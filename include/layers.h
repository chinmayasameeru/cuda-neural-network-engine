// CUDA Neural Network Engine — Layer Definitions
#pragma once

#include "tensor.h"
#include <string>
#include <memory>

namespace cnn {

class Layer {
public:
    virtual ~Layer() = default;
    virtual TensorPtr forward(TensorPtr input) = 0;
    virtual std::vector<TensorPtr> parameters() { return {}; }
    virtual std::string name() const = 0;
    virtual void zero_grad() {
        for (auto& p : parameters())
            if (p) p->zero_grad();
    }
    virtual void train() { training_ = true; }
    virtual void eval() { training_ = false; }

protected:
    bool training_ = true;
    static cublasHandle_t& blas_handle() {
        static cublasHandle_t h = []() {
            cublasHandle_t handle;
            cublasCreate(&handle);
            return handle;
        }();
        return h;
    }
};

using LayerPtr = std::shared_ptr<Layer>;

class Linear : public Layer {
public:
    Linear(int in_features, int out_features, bool use_bias = true);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override {
        return "Linear(" + std::to_string(in_features_) + ", " + std::to_string(out_features_) + ")";
    }
    std::vector<TensorPtr> parameters() override {
        return use_bias_ ? std::vector<TensorPtr>{weight_, bias_} : std::vector<TensorPtr>{weight_};
    }

private:
    int in_features_, out_features_;
    bool use_bias_;
    TensorPtr weight_;
    TensorPtr bias_;
};

class Conv2D : public Layer {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool use_bias = true);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Conv2D"; }
    std::vector<TensorPtr> parameters() override {
        return use_bias_ ? std::vector<TensorPtr>{weight_, bias_} : std::vector<TensorPtr>{weight_};
    }

private:
    int in_c_, out_c_, k_size_, stride_, pad_;
    bool use_bias_;
    TensorPtr weight_;
    TensorPtr bias_;
};

class BatchNorm2D : public Layer {
public:
    BatchNorm2D(int num_features, float eps = 1e-5f, float momentum = 0.1f);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "BatchNorm2D(" + std::to_string(num_features_) + ")"; }
    std::vector<TensorPtr> parameters() override { return {gamma_, beta_}; }

private:
    int num_features_;
    float eps_, momentum_;
    TensorPtr gamma_;
    TensorPtr beta_;
    TensorPtr running_mean_;
    TensorPtr running_var_;
};

class MaxPool2D : public Layer {
public:
    MaxPool2D(int pool_size, int stride = -1);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "MaxPool2D"; }

private:
    int pool_size_, stride_;
};

class AdaptiveAvgPool2D : public Layer {
public:
    AdaptiveAvgPool2D(int output_h, int output_w);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override {
        return "AdaptiveAvgPool2D(" + std::to_string(output_h_) + "x" + std::to_string(output_w_) + ")";
    }

private:
    int output_h_, output_w_;
};

class ReLU : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Sigmoid"; }
};

class Tanh : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Tanh"; }
};

class LeakyReLU : public Layer {
public:
    LeakyReLU(float negative_slope = 0.01f) : negative_slope_(negative_slope) {}
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "LeakyReLU"; }

private:
    float negative_slope_;
};

class GELU : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "GELU"; }
};

class SiLU : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "SiLU"; }
};

class Softmax : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Softmax"; }
};

class Flatten : public Layer {
public:
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Flatten"; }
};

class Dropout : public Layer {
public:
    Dropout(float p = 0.5f) : p_(p) {}
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Dropout(" + std::to_string(p_) + ")"; }

private:
    float p_;
};

} // namespace cnn
