// CUDA Neural Network Engine — Tensor Definition
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

namespace cnn {

class Tensor {
public:
    float* data = nullptr;
    float* grad = nullptr;
    std::vector<int> shape;
    int numel = 0;
    bool requires_grad = false;

    // Autograd
    std::function<void()> backward_fn;
    std::vector<std::shared_ptr<Tensor>> prev;

    Tensor() = default;

    explicit Tensor(const std::vector<int>& shape_, bool requires_grad_ = false);

    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void zero_grad();
    std::vector<float> to_cpu() const;
    void from_cpu(const std::vector<float>& host);

    void backward();
    void print(const std::string& name = "", int max_elems = 16) const;

private:
    int ref_count_ = 1;
};

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace cnn
