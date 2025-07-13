#include "transformers.h"
#include <iostream>

TransformerSelfAttentionLayerImpl::TransformerSelfAttentionLayerImpl(
    MultiHeadAttention attn_,
    MLP mlp_,
    c10::optional<torch::nn::AnyModule> sa_norm_,
    c10::optional<torch::nn::AnyModule> mlp_norm_,
    c10::optional<torch::nn::AnyModule> sa_scale_,
    c10::optional<torch::nn::AnyModule> mlp_scale_
) : attn(attn_), mlp(mlp_) {
    sa_norm = sa_norm_.has_value() ? *sa_norm_ : torch::nn::AnyModule(torch::nn::Identity());
    mlp_norm = mlp_norm_.has_value() ? *mlp_norm_ : torch::nn::AnyModule(torch::nn::Identity());
    sa_scale = sa_scale_.has_value() ? *sa_scale_ : torch::nn::AnyModule(torch::nn::Identity());
    mlp_scale = mlp_scale_.has_value() ? *mlp_scale_ : torch::nn::AnyModule(torch::nn::Identity());

    register_module("attn", attn);
    register_module("mlp", mlp);
    register_module("sa_norm", sa_norm.ptr());
    register_module("mlp_norm", mlp_norm.ptr());
    register_module("sa_scale", sa_scale.ptr());
    register_module("mlp_scale", mlp_scale.ptr());
}

void TransformerSelfAttentionLayerImpl::setup_cache(
    int& batch_size_,
    torch::Dtype& dtype_,
    int& decoder_max_seq_len_) {
    attn->setup_cache(
        batch_size_,
        dtype_,
        decoder_max_seq_len_
    );
}

bool TransformerSelfAttentionLayerImpl::caches_are_setup() const {
    return attn->kv_cache && attn->kv_cache->k_cache.defined();
}

bool TransformerSelfAttentionLayerImpl::caches_are_enabled() const {
    return attn->use_cache;
}

void TransformerSelfAttentionLayerImpl::reset_cache() {
    if (attn->kv_cache) {
        attn->reset_cache();
    }
}

torch::Tensor TransformerSelfAttentionLayerImpl::forward(
    const torch::Tensor& x_,
    const c10::optional<torch::Tensor> mask_,
    const c10::optional<torch::Tensor> input_pos_
) {
    std::cout << "Forwarding TransformerSelfAttentionLayer" << std::endl;
    torch::Tensor h = sa_norm.forward(x_);
    std::cout << "After sa_norm: " << h.sizes() << std::endl;
    auto attn_out = attn->forward(
        h, 
        c10::nullopt, 
        mask_, 
        input_pos_
    );
    std::cout << "attn_out " << attn_out.sizes() << std::endl;
    h = sa_scale.forward(attn_out) + x_;
    std::cout << "h sa_scale " << h.sizes() << std::endl;
    auto mlp_out = mlp->forward(mlp_norm.forward(h));
    return h + mlp_scale.forward(mlp_out);
}

// ---- Sharded Transformer Decoder --- //

// shard_transformer_decoder.cpp
ShardTransformerDecoderImpl::ShardTransformerDecoderImpl(
    const Shard& shard_,
    torch::nn::Embedding tok_embeddings_,
    std::vector<TransformerSelfAttentionLayer> layers_,
    int max_seq_len_,
    RMSNorm norm_,
    torch::nn::Linear output_
) :
    shard(shard_),
    tok_embeddings(tok_embeddings_),
    layers(layers_),
    norm(norm_),
    output(output_)
{
    std::cout << "\nCreating ShardTransformerDecoderImpl with shard: "
              << shard.start_layer << " to " << shard.end_layer << std::endl;
    register_module("tok_embeddings", tok_embeddings);
    register_module("norm", norm);
    register_module("output", output);
    for (size_t i = 0; i < layers.size(); ++i) {
        register_module("layer_" + std::to_string(i), layers[i]);
    }
}

void ShardTransformerDecoderImpl::setup_caches(
    int batch_size_,
    torch::Dtype dtype_,
    int decoder_max_seq_len_
) {
    for (auto& layer : layers) {
        if (layer.ptr() != nullptr) {
            layer->setup_cache(batch_size_, dtype_, decoder_max_seq_len_);
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
    const torch::Tensor& tokens_,
    c10::optional<torch::Tensor> mask_,
    c10::optional<torch::Tensor> input_pos_,
    c10::optional<torch::Tensor> hidden_state_,
    torch::Dtype dtype_
) {
    std::cout << "Forwarding ShardTransformerDecoderImpl" << std::endl;
    torch::Tensor h;
    if (hidden_state_.has_value()) {
        h = hidden_state_.value();
    } else {
        h = tok_embeddings->forward(tokens_).to(dtype_);
    }

    std::cout << "After token embeddings: " << h.sizes() << std::endl;

    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Processing layer " << i << std::endl;
        h = layers[i]->forward(
            h,
            mask_.value_or(torch::Tensor()),
            input_pos_.value_or(torch::Tensor())
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