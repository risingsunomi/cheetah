#pragma once

#include <torch/torch.h>
#include "rope.h"
#include "cache.h"

/**
 * MultiHeadAttentionImpl - Multi-headed attention layer with support for GQA.
 * Inspired by the Torchtune implementation,.
 */
class MultiHeadAttentionImpl : public torch::nn::Module
{
    public:
        MultiHeadAttentionImpl(
            int embed_dim,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            RotaryEmbedding rope,
            bool is_causal = true,
            float attn_dropout = 0.0f);

        void setup_cache(int64_t batch_size, torch::Dtype dtype, int64_t max_seq_len);
        void reset_cache();

        torch::Tensor forward(
            const torch::Tensor &x,
            const c10::optional<torch::Tensor> &y = c10::nullopt,
            const c10::optional<torch::Tensor> &input_pos = c10::nullopt,
            const c10::optional<torch::Tensor> &mask = c10::nullopt);

    private:
        int embed_dim;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int max_seq_len;
        float attn_dropout;
        bool is_causal;
        bool cache_enabled = false;

        RotaryEmbedding rope;
        std::shared_ptr<KVCache> kv_cache;

        torch::nn::Linear q_proj{nullptr};
        torch::nn::Linear k_proj{nullptr};
        torch::nn::Linear v_proj{nullptr};
        torch::nn::Linear out_proj{nullptr};
};

TORCH_MODULE(MultiHeadAttention);
