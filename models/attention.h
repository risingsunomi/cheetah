#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <torch/torch.h>
#include "rope.h"
#include "cache.h"

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(
        int64_t embed_dim,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        torch::nn::Linear q_proj,
        torch::nn::Linear k_proj,
        torch::nn::Linear v_proj,
        torch::nn::Linear out_proj,
        std::shared_ptr<RotaryEmbedding> pos_emb = nullptr,
        std::shared_ptr<KVCache> kv_cache = nullptr,
        bool is_causal = true,
        double attn_dropout = 0.0,
        bool is_cache_enabled = false
    );

    std::shared_ptr<KVCache> kv_cache;
    bool is_cache_enabled;
    void setup_cache(
        int64_t batch_size,
        torch::Dtype dtype,
        int64_t max_seq_len = 2048
    );
    void reset_cache();

    torch::Tensor forward(
        torch::Tensor x,
        c10::optional<torch::Tensor> y = c10::nullopt,
        c10::optional<torch::Tensor> mask = c10::nullopt,
        c10::optional<torch::Tensor> input_pos = c10::nullopt);

private:
    int64_t embed_dim;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t max_seq_len;
    double attn_dropout;
    bool is_causal;
    

    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};

    std::shared_ptr<RotaryEmbedding> pos_emb;
    
};

TORCH_MODULE(MultiHeadAttention);

#endif // MULTI_HEAD_ATTENTION_H