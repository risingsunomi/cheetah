#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include <torch/torch.h>
#include <vector>
#include <memory>
#include "attention.h"
#include "mlp.h"
#include "rms.h"
#include "utils/cache.h"
#include "utils/shard.h"

class TransformerSelfAttentionLayerImpl : public torch::nn::Module {
public:
    TransformerSelfAttentionLayerImpl(
        MultiHeadAttention attn,
        MLP mlp,
        c10::optional<torch::nn::AnyModule> sa_norm_ = c10::nullopt,
        c10::optional<torch::nn::AnyModule> mlp_norm_ = c10::nullopt,
        c10::optional<torch::nn::AnyModule> sa_scale_ = c10::nullopt,
        c10::optional<torch::nn::AnyModule> mlp_scale_ = c10::nullopt
    );

    torch::Tensor forward(
        const torch::Tensor& x,
        const c10::optional<torch::Tensor>& mask = c10::nullopt,
        const c10::optional<torch::Tensor>& input_pos = c10::nullopt
    );

    void setup_cache(int64_t batch_size, torch::Dtype dtype, int64_t decoder_max_seq_len);
    bool caches_are_setup() const;
    bool caches_are_enabled() const;
    void reset_cache();

private:
    MultiHeadAttention attn;
    MLP mlp;
    torch::nn::AnyModule sa_norm;
    torch::nn::AnyModule mlp_norm;
    torch::nn::AnyModule sa_scale;
    torch::nn::AnyModule mlp_scale;
};

TORCH_MODULE(TransformerSelfAttentionLayer);

// ---- Sharded Transformer Decoder --- //
class ShardTransformerDecoderImpl : public torch::nn::Module {
public:
    ShardTransformerDecoderImpl(
        const Shard& shard_,
        torch::nn::Embedding tok_embeddings_,
        std::vector<TransformerSelfAttentionLayer> layers_,
        int64_t max_seq_len_,
        RMSNorm norm_,
        torch::nn::Linear output_
    );

    void setup_caches(
        int64_t batch_size,
        torch::Dtype dtype,
        c10::optional<int64_t> decoder_max_seq_len = c10::nullopt
    );

    bool caches_are_enabled() const;
    void reset_caches();

    torch::Tensor forward(
        const torch::Tensor& tokens,
        c10::optional<torch::Tensor> mask = c10::nullopt,
        c10::optional<torch::Tensor> input_pos = c10::nullopt,
        c10::optional<torch::Tensor> hidden_state = c10::nullopt,
        torch::Dtype dtype = torch::kBFloat16
    );

private:
    Shard shard;
    torch::nn::Embedding tok_embeddings{nullptr};
    std::vector<TransformerSelfAttentionLayer> layers;
    RMSNorm norm{nullptr};
    torch::nn::Linear output{nullptr};
    int64_t max_seq_len;
    int64_t decoder_max_cache_seq_len = -1;
};
TORCH_MODULE(ShardTransformerDecoder);

#endif // TRANSFORMER_H