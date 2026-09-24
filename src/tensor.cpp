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
        data = static_cast<float*>(MemoryArena::instance().allocate(numel * sizeof(float)));
        if (requires_grad) {
            grad = static_cast<float*>(MemoryArena::instance().allocate(numel * sizeof(float)));
            cudaMemset(grad, 0, numel * sizeof(float));
        }
    }
}

Tensor::~Tensor() {
    if (data) {
        MemoryArena::instance().deallocate(data, numel * sizeof(float));
        data = nullptr;
    }
    if (grad) {
        MemoryArena::instance().deallocate(grad, numel * sizeof(float));
        grad = nullptr;
    }
}

Tensor::Tensor(Tensor&& o) noexcept
    : data(o.data), grad(o.grad), shape(std::move(o.shape)),
      numel(o.numel), requires_grad(o.requires_grad),
      backward_fn(std::move(o.backward_fn)), prev(std::move(o.prev))
{
    o.data = nullptr;
    o.grad = nullptr;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        this->~Tensor();
        data = o.data; grad = o.grad;
        shape = std::move(o.shape);
        numel = o.numel; requires_grad = o.requires_grad;
        backward_fn = std::move(o.backward_fn); prev = std::move(o.prev);
        o.data = nullptr; o.grad = nullptr;
    }
    return *this;
}

void Tensor::zero_grad() {
    if (grad) cudaMemset(grad, 0, numel * sizeof(float));
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

float Tensor::item() const {
    if (numel != 1) throw std::runtime_error("item() only for scalar tensors");
    float v;
    cudaMemcpy(&v, data, sizeof(float), cudaMemcpyDeviceToHost);
    return v;
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
