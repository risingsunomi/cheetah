#include "transformer.h"
#include <iostream>

TransformerSelfAttentionLayerImpl::TransformerSelfAttentionLayerImpl(
    MultiHeadAttention attn,
    MLP mlp,
    torch::nn::AnyModule sa_norm,
    torch::nn::AnyModule mlp_norm,
    c10::optional<torch::nn::AnyModule> sa_scale,
    c10::optional<torch::nn::AnyModule> mlp_scale
) : attn(attn), mlp(mlp) {
    sa_norm = sa_norm.is_empty() ? torch::nn::AnyModule(torch::nn::Identity()) : sa_norm;
    mlp_norm = mlp_norm.is_empty() ? torch::nn::AnyModule(torch::nn::Identity()) : mlp_norm;
    sa_scale = sa_scale ? torch::nn::AnyModule(torch::nn::Identity()) : sa_scale;
    mlp_scale = mlp_scale ? torch::nn::AnyModule(torch::nn::Identity()) : mlp_scale;

    register_module("attn", attn);
    register_module("mlp", mlp);
    register_module("sa_norm", sa_norm.ptr());
    register_module("mlp_norm", mlp_norm.ptr());
    register_module("sa_scale", sa_scale->ptr()); // Always register
    register_module("mlp_scale", mlp_scale->ptr()); // Always register
}

void TransformerSelfAttentionLayerImpl::setup_cache(
    int64_t batch_size,
    torch::Dtype dtype,
    int64_t decoder_max_seq_len) {
    attn->setup_cache(batch_size, dtype, decoder_max_seq_len);
}

bool TransformerSelfAttentionLayerImpl::caches_are_setup() const {
    return attn->kv_cache && attn->kv_cache->k_cache.defined();
}

bool TransformerSelfAttentionLayerImpl::caches_are_enabled() const {
    return attn->is_cache_enabled;
}

void TransformerSelfAttentionLayerImpl::reset_cache() {
    if (attn->kv_cache) {
        attn->reset_cache();
    }
}

torch::Tensor TransformerSelfAttentionLayerImpl::forward(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& mask,
    const c10::optional<torch::Tensor>& input_pos
) {
    torch::Tensor h = sa_norm.forward(x);
    auto attn_out = attn->forward(h, h, mask, input_pos);
    h = sa_scale.forward(attn_out) + x;
    auto mlp_out = mlp->forward(mlp_norm.forward(h));
    return h + mlp_scale.forward(mlp_out);
}

// ---- Sharded Transformer Decoder --- //

// shard_transformer_decoder.cpp
ShardTransformerDecoderImpl::ShardTransformerDecoderImpl(
    const Shard& shard,
    torch::nn::Embedding tok_embeddings,
    std::vector<TransformerSelfAttentionLayer> layers,
    int64_t max_seq_len,
    int64_t num_heads,
    int64_t head_dim,
    RMSNorm norm,
    torch::nn::Linear output
) :
    shard(shard),
    tok_embeddings(tok_embeddings),
    layers(std::move(layers)),
    norm(norm),
    output(output),
    max_seq_len(max_seq_len)
{
    register_module("tok_embeddings", tok_embeddings);
    register_module("norm", norm);
    register_module("output", output);
    for (size_t i = 0; i < layers.size(); ++i) {
        register_module("layer_" + std::to_string(i), layers[i]);
    }
}

void ShardTransformerDecoderImpl::setup_caches(
    int64_t batch_size,
    torch::Dtype dtype,
    c10::optional<int64_t> decoder_max_seq_len
) {
    decoder_max_cache_seq_len = decoder_max_seq_len.value_or(max_seq_len);
    for (auto& layer : layers) {
        if (layer.ptr() != nullptr) {
            layer->setup_cache(batch_size, dtype, decoder_max_cache_seq_len);
        }
    }
}

bool ShardTransformerDecoderImpl::caches_are_enabled() const {
    for (const auto& layer : layers) {
        if (layer->caches_are_enabled()) {
            return true;
        }
    }
    return false;
}

void ShardTransformerDecoderImpl::reset_caches() {
    for (auto& layer : layers) {
        layer->reset_cache();
    }
}

torch::Tensor ShardTransformerDecoderImpl::forward(
    const torch::Tensor& tokens,
    c10::optional<torch::Tensor> mask,
    c10::optional<torch::Tensor> input_pos,
    c10::optional<torch::Tensor> hidden_state,
    torch::Dtype dtype
) {
    torch::Tensor h;
    if (hidden_state.has_value()) {
        h = hidden_state.value();
    } else {
        h = tok_embeddings->forward(tokens).to(dtype);
    }

    for (int i = shard.start_layer; i <= shard.end_layer; ++i) {
        h = layers[i]->forward(
            h,
            mask.value_or(torch::Tensor()),
            input_pos.value_or(torch::Tensor())
        );
    }

    if (shard.is_last_layer()) {
        h = norm->forward(h);
        auto out = output->forward(h);
        return out;
    } else {
        return h;
    }
}