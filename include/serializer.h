// CUDA Neural Network Engine — Model Serialization (v0.2.0)
#pragma once

#include "model.h"
#include <vector>
#include <string>

namespace cnn {

// ============================================================
// Model Save/Load (NPZ-like format)
// ============================================================
class ModelSerializer {
public:
    // Save model weights to a file
    static void save(const std::vector<TensorPtr>& params, const std::string& filename);

    // Load model weights from a file
    static std::vector<TensorPtr> load(const std::string& filename);

private:
    // Binary format: [num_params][shape_data...][weight_data...]
    static constexpr uint32_t MAGIC = 0x434E4E50;  // "CNNP"
};

} // namespace cnn
