// CUDA Neural Network Engine — Tensor Implementation
#include "tensor.h"
#include <algorithm>
#include <cstring>

namespace cnn {

Tensor::Tensor(const std::vector<int>& shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_)
{
    numel = 1;
    for (int s : shape) numel *= s;

    if (numel > 0) {
        cudaError_t err = cudaMalloc(&data, numel * sizeof(float));
        if (err != cudaSuccess)
            throw std::runtime_error("Failed to allocate GPU memory for tensor data");

        if (requires_grad) {
            err = cudaMalloc(&grad, numel * sizeof(float));
            if (err != cudaSuccess) {
                cudaFree(data);
                throw std::runtime_error("Failed to allocate GPU memory for tensor grad");
            }
            cudaMemset(grad, 0, numel * sizeof(float));
        }
    }
}

Tensor::~Tensor() {
    if (data) {
        cudaFree(data);
        data = nullptr;
    }
    if (grad) {
        cudaFree(grad);
        grad = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data), grad(other.grad),
      shape(std::move(other.shape)), numel(other.numel),
      requires_grad(other.requires_grad),
      backward_fn(std::move(other.backward_fn)),
      prev(std::move(other.prev))
{
    other.data = nullptr;
    other.grad = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        this->~Tensor();
        data = other.data;
        grad = other.grad;
        shape = std::move(other.shape);
        numel = other.numel;
        requires_grad = other.requires_grad;
        backward_fn = std::move(other.backward_fn);
        prev = std::move(other.prev);
        other.data = nullptr;
        other.grad = nullptr;
    }
    return *this;
}

void Tensor::zero_grad() {
    if (grad) {
        cudaMemset(grad, 0, numel * sizeof(float));
    }
}

std::vector<float> Tensor::to_cpu() const {
    std::vector<float> host(numel);
    if (data && numel > 0)
        cudaMemcpy(host.data(), data, numel * sizeof(float), cudaMemcpyDeviceToHost);
    return host;
}

void Tensor::from_cpu(const std::vector<float>& host) {
    if (data && host.size() == (size_t)numel)
        cudaMemcpy(data, host.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::backward() {
    if (!requires_grad) return;
    if (backward_fn) backward_fn();
    for (auto& p : prev) {
        if (p && p->requires_grad) p->backward();
    }
}

void Tensor::print(const std::string& name, int max_elems) const {
    auto h = to_cpu();
    std::cout << name << " [";
    for (int i = 0; i < std::min(numel, max_elems); ++i) {
        std::cout << h[i];
        if (i < std::min(numel, max_elems) - 1) std::cout << ", ";
    }
    if (numel > max_elems) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

} // namespace cnn
