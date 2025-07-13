// General Multi-Head Attention Model
// Generalized model for use with LLMs and other multi-head attention models
#ifndef GENERAL_MHA_MODEL_H
#define GENERAL_MHA_MODEL_H

#include <torch/torch.h>
#include "rope.h"
#include "attention.h"
#include "utils/cache.h"
#include "rms.h"
#include "mlp.h"
#include "transformers.h"
#include "utils/shard.h"

class GeneralMHAModel : public torch::nn::Module {
    public:
        GeneralMHAModel(
            const Shard& shard_,
            int vocab_size_,
            int embed_dim_,
            int hidden_dim_,
            int num_heads_,
            int num_kv_heads_,
            int head_dim_,
            int max_seq_len_,
            float_t rope_scaling_,
            bool is_cache_enabled = false
        );
        torch::Tensor forward(
            const torch::Tensor& tokens,
            const c10::optional<torch::Tensor>& mask,
            const c10::optional<torch::Tensor>& input_pos,
            const c10::optional<torch::Tensor>& hidden_state
        );

        bool is_cache_enabled;
        std::vector<TransformerSelfAttentionLayer> self_attn_layers;
        ShardTransformerDecoder shard_decoder = nullptr;

    private:
        const Shard& shard;
        int vocab_size;
        int embed_dim;
        int hidden_dim;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int max_seq_len;
        float_t rope_scaling;
        bool is_cache_enabled;
};

#endif // GENERAL_MHA_MODEL_H