// General Multi-Head Attention Model
// Generalized model for use with LLMs and other multi-head attention models
#ifndef GENERAL_MHA_MODEL_H
#define GENERAL_MHA_MODEL_H

#include <torch/torch.h>
#include "rope.h"
#include "attention.h"
#include "cache.h"
#include "rms.h"
#include "mlp.h"
#include "transformer.h"
#include "shard.h"

class GeneralMHAModel : public torch::nn::Module {
    public:
        GeneralMHAModel(
            const Shard& shard_,
            int64_t vocab_size,
            int64_t embed_dim,
            int64_t hidden_dim,
            int64_t num_heads,
            int64_t num_kv_heads,
            int64_t head_dim,
            int64_t max_seq_len,
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
};

#endif // GENERAL_MHA_MODEL_H