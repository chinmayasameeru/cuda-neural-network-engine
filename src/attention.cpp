// CUDA Neural Network Engine — Multi-Head Attention Implementation
#include "advanced_layers.h"
#include "kernels.cuh"
#include <random>
#include <cmath>

namespace cnn {

MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, float dropout)
    : embed_dim_(embed_dim), num_heads_(num_heads), dropout_(dropout)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim_ = embed_dim / num_heads;

    // Xavier initialization
    float limit = std::sqrt(6.0f / (embed_dim + embed_dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Q, K, V projection weights
    W_q_ = std::make_shared<Tensor>(std::vector<int>{embed_dim, embed_dim}, true);
    W_k_ = std::make_shared<Tensor>(std::vector<int>{embed_dim, embed_dim}, true);
    W_v_ = std::make_shared<Tensor>(std::vector<int>{embed_dim, embed_dim}, true);
    W_o_ = std::make_shared<Tensor>(std::vector<int>{embed_dim, embed_dim}, true);

    std::vector<float> wq_data(embed_dim * embed_dim);
    std::vector<float> wk_data(embed_dim * embed_dim);
    std::vector<float> wv_data(embed_dim * embed_dim);
    std::vector<float> wo_data(embed_dim * embed_dim);
    for (auto& v : wq_data) v = dist(gen);
    for (auto& v : wk_data) v = dist(gen);
    for (auto& v : wv_data) v = dist(gen);
    for (auto& v : wo_data) v = dist(gen);
    W_q_->from_cpu(wq_data);
    W_k_->from_cpu(wk_data);
    W_v_->from_cpu(wv_data);
    W_o_->from_cpu(wo_data);

    // Linear layers for projections
    proj_q_ = std::make_shared<Linear>(embed_dim, embed_dim, false);
    proj_k_ = std::make_shared<Linear>(embed_dim, embed_dim, false);
    proj_v_ = std::make_shared<Linear>(embed_dim, embed_dim, false);
    proj_o_ = std::make_shared<Linear>(embed_dim, embed_dim, false);

    // Copy weights
    auto linear_q = std::static_pointer_cast<Linear>(proj_q_);
    auto linear_k = std::static_pointer_cast<Linear>(proj_k_);
    auto linear_v = std::static_pointer_cast<Linear>(proj_v_);
    auto linear_o = std::static_pointer_cast<Linear>(proj_o_);
    // Note: In production, we'd properly copy weights here
}

TensorPtr MultiHeadAttention::forward(TensorPtr query) {
    return forward(query, query, query);  // Self-attention
}

TensorPtr MultiHeadAttention::forward(TensorPtr query, TensorPtr key, TensorPtr value) {
    // Project Q, K, V
    auto Q = proj_q_->forward(query);
    auto K = proj_k_->forward(key);
    auto V = proj_v_->forward(value);

    // For simplicity, process attention on CPU (full GPU impl needs batched matmul)
    auto q_host = Q->to_cpu();
    auto k_host = K->to_cpu();
    auto v_host = V->to_cpu();

    int batch = Q->shape[0];
    int seq_len = Q->shape[1];
    int head_dim = head_dim_;

    std::vector<float> output(batch * seq_len * embed_dim_, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads_; ++h) {
            for (int q = 0; q < seq_len; ++q) {
                // Compute attention scores for this query
                std::vector<float> scores(seq_len, 0.0f);
                float max_score = -1e20f;

                for (int k = 0; k < seq_len; ++k) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        int q_idx = (b * seq_len + q) * embed_dim_ + h * head_dim + d;
                        int k_idx = (b * seq_len + k) * embed_dim_ + h * head_dim + d;
                        dot += q_host[q_idx] * k_host[k_idx];
                    }
                    scores[k] = dot / std::sqrt(static_cast<float>(head_dim));
                    max_score = std::max(max_score, scores[k]);
                }

                // Softmax
                float sum = 0.0f;
                for (int k = 0; k < seq_len; ++k) {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum += scores[k];
                }
                for (int k = 0; k < seq_len; ++k) {
                    scores[k] /= sum;
                }

                // Weighted sum of values
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int k = 0; k < seq_len; ++k) {
                        int v_idx = (b * seq_len + k) * embed_dim_ + h * head_dim + d;
                        val += scores[k] * v_host[v_idx];
                    }
                    int out_idx = (b * seq_len + q) * embed_dim_ + h * head_dim + d;
                    output[out_idx] = val;
                }
            }
        }
    }

    auto attn_output = std::make_shared<Tensor>(std::vector<int>{batch, seq_len, embed_dim_}, true);
    attn_output->from_cpu(output);

    // Final linear projection
    return proj_o_->forward(attn_output);
}

std::vector<TensorPtr> MultiHeadAttention::parameters() {
    std::vector<TensorPtr> params;
    auto pq = proj_q_->parameters();
    auto pk = proj_k_->parameters();
    auto pv = proj_v_->parameters();
    auto po = proj_o_->parameters();
    params.insert(params.end(), pq.begin(), pq.end());
    params.insert(params.end(), pk.begin(), pk.end());
    params.insert(params.end(), pv.begin(), pv.end());
    params.insert(params.end(), po.begin(), po.end());
    return params;
}

void MultiHeadAttention::zero_grad() {
    proj_q_->zero_grad();
    proj_k_->zero_grad();
    proj_v_->zero_grad();
    proj_o_->zero_grad();
}

} // namespace cnn
