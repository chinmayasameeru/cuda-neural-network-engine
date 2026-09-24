// CUDA Neural Network Engine — Advanced Layers (v0.2.0)
#pragma once

#include "tensor.h"
#include "layers.h"
#include <string>
#include <memory>
#include <vector>
#include <cmath>

namespace cnn {

// ============================================================
// Residual Block (for ResNet architectures)
// ============================================================
class ResidualBlock : public Layer {
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1);

    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "ResidualBlock(" + std::to_string(in_c_) + "," + std::to_string(out_c_) + ")"; }
    std::vector<TensorPtr> parameters() override;
    void zero_grad() override;

private:
    int in_c_, out_c_, stride_;
    LayerPtr conv1_, bn1_, relu1_;
    LayerPtr conv2_, bn2_, relu2_;
    LayerPtr shortcut_;
    bool use_shortcut_;
};

// ============================================================
// LSTM Cell
// ============================================================
class LSTMCell : public Layer {
public:
    LSTMCell(int input_size, int hidden_size);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "LSTMCell(" + std::to_string(in_size_) + "," + std::to_string(hidden_size_) + ")"; }
    std::vector<TensorPtr> parameters() override;
    void zero_grad() override;

    void reset_state();

private:
    int in_size_, hidden_size_;
    TensorPtr Wf_, Wi_, Wc_, Wo_;  // Weight matrices
    TensorPtr bf_, bi_, bc_, bo_;  // Biases
    TensorPtr h_prev_, c_prev_;    // State
    bool first_call_ = true;
};

// ============================================================
// GRU Cell
// ============================================================
class GRUCell : public Layer {
public:
    GRUCell(int input_size, int hidden_size);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "GRUCell(" + std::to_string(in_size_) + "," + std::to_string(hidden_size_) + ")"; }
    std::vector<TensorPtr> parameters() override;
    void zero_grad() override;

    void reset_state();

private:
    int in_size_, hidden_size_;
    TensorPtr Wz_, Wr_, Wh_;  // Weight matrices
    TensorPtr bz_, br_, bh_;  // Biases
    TensorPtr h_prev_;        // State
    bool first_call_ = true;
};

// ============================================================
// Multi-Head Attention (Transformer component)
// ============================================================
class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(int embed_dim, int num_heads, float dropout = 0.0f);
    TensorPtr forward(TensorPtr query) override;
    TensorPtr forward(TensorPtr query, TensorPtr key, TensorPtr value);
    std::string name() const override { return "MultiHeadAttention(" + std::to_string(embed_dim_) + "," + std::to_string(num_heads_) + ")"; }
    std::vector<TensorPtr> parameters() override;
    void zero_grad() override;

private:
    int embed_dim_, num_heads_, head_dim_;
    LayerPtr W_q_, W_k_, W_v_, W_o_;
    LayerPtr proj_q_, proj_k_, proj_v_, proj_o_;
    float dropout_;
};

// ============================================================
// Layer Normalization
// ============================================================
class LayerNorm : public Layer {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5f);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "LayerNorm(" + std::to_string(normalized_shape_) + ")"; }
    std::vector<TensorPtr> parameters() override;

private:
    int normalized_shape_;
    float eps_;
    TensorPtr gamma_;
    TensorPtr beta_;
};

// ============================================================
// 1D Convolution (for sequences)
// ============================================================
class Conv1D : public Layer {
public:
    Conv1D(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0, bool use_bias = true);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Conv1D"; }
    std::vector<TensorPtr> parameters() override {
        return use_bias_ ? std::vector<TensorPtr>{weight_, bias_} : std::vector<TensorPtr>{weight_};
    }

private:
    int in_c_, out_c_, k_size_, stride_, pad_;
    bool use_bias_;
    TensorPtr weight_;
    TensorPtr bias_;
};

// ============================================================
// Upsampling / Transposed Convolution
// ============================================================
class ConvTranspose2D : public Layer {
public:
    ConvTranspose2D(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 0, bool use_bias = true);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "ConvTranspose2D"; }
    std::vector<TensorPtr> parameters() override {
        return use_bias_ ? std::vector<TensorPtr>{weight_, bias_} : std::vector<TensorPtr>{weight_};
    }

private:
    int in_c_, out_c_, k_size_, stride_, pad_;
    bool use_bias_;
    TensorPtr weight_;
    TensorPtr bias_;
};

// ============================================================
// Embedding Layer
// ============================================================
class Embedding : public Layer {
public:
    Embedding(int num_embeddings, int embedding_dim);
    TensorPtr forward(TensorPtr input) override;
    std::string name() const override { return "Embedding(" + std::to_string(num_emb_) + "," + std::to_string(emb_dim_) + ")"; }
    std::vector<TensorPtr> parameters() override { return {weight_}; }

private:
    int num_emb_, emb_dim_;
    TensorPtr weight_;
};

} // namespace cnn
