#include "general_mha_model.h"
#include "transformer.h"
#include "attention.h"

GeneralMHAModel::GeneralMHAModel(
    const Shard& shard_,
    int64_t vocab_size,
    int64_t embed_dim,
    int64_t hidden_dim,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_seq_len,
    bool is_cache_enabled
) : shard(shard_),
    is_cache_enabled(is_cache_enabled) {

    // add grabbing informaiton from model json config

    // Create decoder layers from layer_start to layer_end
    for (int64_t i = shard.start_layer; i < shard.end_layer; ++i) {
        // Instantiate TransformerSelfAttentionLayer
        auto transformer_layer = register_module(
            "transformer",
            TransformerSelfAttentionLayer(
                MultiHeadAttention(
                    embed_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    torch::nn::Linear(embed_dim, embed_dim), // q_proj
                    torch::nn::Linear(embed_dim, embed_dim), // k_proj
                    torch::nn::Linear(embed_dim, embed_dim), // v_proj
                    torch::nn::Linear(embed_dim, embed_dim), // out_proj
                    std::make_shared<RotaryEmbedding>(
                        head_dim, 
                        max_seq_len
                    ),
                    nullptr, // kv_cache will be set up later if needed
                    true,    // is_causal
                    0.0,     // attn_dropout
                    false    // is_cache_enabled, set later if needed
                ),
                MLP(
                    embed_dim,
                    hidden_dim
                ),
                torch::nn::AnyModule(RMSNorm(embed_dim)),
                torch::nn::AnyModule(RMSNorm(embed_dim))
            )
        );
        
        self_attn_layers.push_back(
            register_module("layer_" + std::to_string(i), transformer_layer));
    }

    shard_decoder = ShardTransformerDecoder(
        shard,
        torch::nn::Embedding(vocab_size, embed_dim),
        self_attn_layers,
        max_seq_len,
        num_heads,
        head_dim,
        RMSNorm(embed_dim),
        torch::nn::Linear(embed_dim, vocab_size)
    );
}

torch::Tensor GeneralMHAModel::forward(
    const torch::Tensor& tokens,
    const c10::optional<torch::Tensor>& mask,
    const c10::optional<torch::Tensor>& input_pos,
    const c10::optional<torch::Tensor>& hidden_state
) {
    return shard_decoder->forward(tokens, mask, input_pos, hidden_state);
}
