// CUDA Neural Network Engine — Model Serialization
#include "serializer.h"
#include <fstream>
#include <cstring>

namespace cnn {

void ModelSerializer::save(const std::vector<TensorPtr>& params, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write magic number
    uint32_t magic = MAGIC;
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));

    // Write number of parameters
    uint32_t num_params = params.size();
    file.write(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    for (const auto& p : params) {
        if (!p) continue;

        // Write shape info
        uint32_t ndim = p->shape.size();
        file.write(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        for (int s : p->shape) {
            int32_t dim = s;
            file.write(reinterpret_cast<char*>(&dim), sizeof(dim));
        }

        // Write data
        auto cpu_data = p->to_cpu();
        uint64_t num_bytes = cpu_data.size() * sizeof(float);
        file.write(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));
        file.write(reinterpret_cast<char*>(cpu_data.data()), num_bytes);
    }

    file.close();
}

std::vector<TensorPtr> ModelSerializer::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Read magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MAGIC) {
        throw std::runtime_error("Invalid model file format");
    }

    // Read number of parameters
    uint32_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    std::vector<TensorPtr> params;
    for (uint32_t i = 0; i < num_params; ++i) {
        // Read shape info
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int> shape(ndim);
        for (uint32_t j = 0; j < ndim; ++j) {
            int32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            shape[j] = dim;
        }

        // Read data
        uint64_t num_bytes;
        file.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));
        std::vector<float> data(num_bytes / sizeof(float));
        file.read(reinterpret_cast<char*>(data.data()), num_bytes);

        auto t = std::make_shared<Tensor>(shape, true);
        t->from_cpu(data);
        params.push_back(t);
    }

    file.close();
    return params;
}

} // namespace cnn
