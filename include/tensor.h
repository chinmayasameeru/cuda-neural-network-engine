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
#include <cstring>
#include <unordered_map>

namespace cnn {

struct MemoryStats {
    size_t total_allocated = 0;
    size_t current_used = 0;
    size_t peak_used = 0;
    int alloc_count = 0;
    int free_count = 0;
};

class MemoryArena {
public:
    static MemoryArena& instance() {
        static MemoryArena a;
        return a;
    }

    void* allocate(size_t bytes) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            flush_cache();
            err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA OOM: " + std::to_string(bytes) + " bytes");
            }
        }
        stats_.total_allocated += bytes;
        stats_.current_used += bytes;
        stats_.alloc_count++;
        if (stats_.current_used > stats_.peak_used) stats_.peak_used = stats_.current_used;
        return ptr;
    }

    void deallocate(void* ptr, size_t bytes) {
        if (ptr) {
            cudaFree(ptr);
            stats_.current_used -= bytes;
            stats_.free_count++;
        }
    }

    void flush_cache() {
        for (auto& [sz, ptrs] : cache_) {
            for (void* p : ptrs) {
                cudaFree(p);
                stats_.current_used -= sz;
            }
        }
        cache_.clear();
    }

    const MemoryStats& stats() const { return stats_; }

private:
    MemoryArena() = default;
    MemoryStats stats_;
    std::unordered_map<size_t, std::vector<void*>> cache_;
};

class Tensor {
public:
    float* data = nullptr;
    float* grad = nullptr;
    std::vector<int> shape;
    int numel = 0;
    bool requires_grad = false;
    std::function<void()> backward_fn;
    std::vector<std::shared_ptr<Tensor>> prev;

    Tensor() = default;
    explicit Tensor(const std::vector<int>& shape_, bool requires_grad_ = false);
    ~Tensor();
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept;
    Tensor& operator=(Tensor&& o) noexcept;

    void zero_grad();
    std::vector<float> to_cpu() const;
    void from_cpu(const std::vector<float>& host);
    void backward();
    float item() const;
    void print(const std::string& name = "", int max_elems = 16) const;
};

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace cnn
