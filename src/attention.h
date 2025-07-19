#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <torch/torch.h>
#include "rope.h"
#include "utils/cache.h"

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(
        int layer_id_,
        int embed_dim_,
        int num_heads_,
        int num_kv_heads_,
        int head_dim_,
        torch::nn::Linear q_proj_,
        torch::nn::Linear k_proj_,
        torch::nn::Linear v_proj_,
        torch::nn::Linear out_proj_,
        RotaryEmbedding pos_emb_,
        c10::optional<KVCache> kv_cache_,
        bool is_causal_,
        float attn_dropout_,
        bool use_cache_,
        const c10::ScalarType& model_dtype_
    );

    void setup_cache(
        int batch_size_,
        int max_seq_len_
    );
    void reset_cache();

    torch::Tensor forward(
        torch::Tensor& x_,
        c10::optional<torch::Tensor> y_,
        c10::optional<torch::Tensor> mask_,
        c10::optional<torch::Tensor> input_pos_);

    c10::optional<KVCache> kv_cache = c10::nullopt;
    bool use_cache;
    bool cache_enabled;

private:
    int layer_id;
    int embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    bool is_causal;
    float attn_dropout;
    const c10::ScalarType model_dtype;
    
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};

    RotaryEmbedding pos_emb;
    
};

TORCH_MODULE(MultiHeadAttention);

#endif // MULTI_HEAD_ATTENTION_H