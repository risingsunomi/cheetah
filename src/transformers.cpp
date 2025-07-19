#include "transformers.h"
#include <iostream>

TransformerSelfAttentionLayerImpl::TransformerSelfAttentionLayerImpl(
  int layer_idx_,
  MultiHeadAttention& attn_,
  MLP& mlp_,
  c10::optional<torch::nn::AnyModule> input_layernorm_,
  c10::optional<torch::nn::AnyModule> post_attn_layernorm_,
  c10::optional<torch::nn::AnyModule> input_scale_,
  c10::optional<torch::nn::AnyModule> post_attn_scale_,
  const c10::ScalarType& model_dtype_
) : attn(attn_),
  mlp(mlp_),
  model_dtype(model_dtype_),
  layer_idx(layer_idx_) {

  input_layernorm = input_layernorm_.has_value() ? *input_layernorm_ : torch::nn::AnyModule(torch::nn::Identity());
  
  post_attn_layernorm = post_attn_layernorm_.has_value() ? *post_attn_layernorm_ : torch::nn::AnyModule(torch::nn::Identity());
  
  input_scale = input_scale_.has_value() ? *input_scale_ : torch::nn::AnyModule(torch::nn::Identity());
  
  post_attn_scale = post_attn_scale_.has_value() ? *post_attn_scale_ : torch::nn::AnyModule(torch::nn::Identity());

  register_module(
    "model__layers__" +
    std::to_string(layer_idx) +
    "__self_attn",
    attn
  );
  
  register_module(
    "model__layers__" +
    std::to_string(layer_idx) +
    "__mlp",
    mlp
  );

  register_module(
    "model__layers__" +
    std::to_string(layer_idx) +
    "__input_layernorm",
    input_layernorm.ptr()
  );

  register_module(
    "model__layers__" +
    std::to_string(layer_idx) +
    "__post_attn_layernorm",
    post_attn_layernorm.ptr()
  );

  register_module(
    "model__layers__" +
    std::to_string(layer_idx) +
    "__input_scale",
    input_scale.ptr()
  );

  register_module(
    "model__layers__" +
    std::to_string(layer_idx) +
    "__post_attn_scale",
    post_attn_scale.ptr()
  );
}

void TransformerSelfAttentionLayerImpl::setup_cache(
    int& batch_size_,
    int& decoder_max_seq_len_) {
    attn->setup_cache(
        batch_size_,
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
    torch::Tensor h = input_layernorm.forward(x_);
    std::cout << "After input_layernorm: " << h.sizes() << std::endl;
    std::cout << h.dtype().name() << std::endl;
    std::cout << "Mask " << mask_->sizes() << "\\" << mask_->dtype().name() << std::endl;
    std::cout << "Input POS " << input_pos_->sizes() << "\\" << input_pos_->dtype().name() << std::endl;
    auto attn_out = attn->forward(
        h, 
        c10::nullopt, 
        mask_, 
        input_pos_
    );
    std::cout << "attn_out " << attn_out.sizes() << std::endl;
    auto input_scale_out = input_scale.forward(attn_out);
    h = input_scale_out + x_;
    std::cout << "h input_scale " << h.sizes() << std::endl;
    std::cout << h.dtype().name() << std::endl;
    auto post_attn_layernorm_out = post_attn_layernorm.forward(h);
    std::cout << "post_attn_layernorm_out " << post_attn_layernorm_out.sizes() << std::endl;
    std::cout << "post_attn_layernorm_out " << post_attn_layernorm_out.dtype().name() << std::endl;
    auto mlp_out = mlp->forward(post_attn_layernorm_out);
    std::cout << "mlp_out " << mlp_out.sizes() << std::endl;
    std::cout << "mlp_out " << mlp_out.dtype().name() << std::endl;
    return h + post_attn_scale.forward(mlp_out);
}

// ---- Sharded Transformer Decoder --- //

// shard_transformer_decoder.cpp
ShardTransformerDecoderImpl::ShardTransformerDecoderImpl(
    const Shard& shard_,
    torch::nn::Embedding tok_embeddings_,
    std::vector<TransformerSelfAttentionLayer> layers_,
    int max_seq_len_,
    RMSNorm norm_,
    torch::nn::Linear output_,
    const c10::ScalarType &dtype_
) :
    shard(shard_),
    tok_embeddings(tok_embeddings_),
    layers(layers_),
    norm(norm_),
    output(output_),
    model_dtype(dtype_)
{
    std::cout << "\nCreating ShardTransformerDecoderImpl with shard: "
              << shard.start_layer << " to " << shard.end_layer << std::endl;
    register_module("model__embed_tokens", tok_embeddings);
    register_module("model__norm", norm);
    register_module("model__lm_head", output);
}

void ShardTransformerDecoderImpl::setup_caches(
    int batch_size_,
    int decoder_max_seq_len_
) {
    for (auto& layer : layers) {
        if (!layer.is_empty()) {
            layer->setup_cache(
                batch_size_,
                decoder_max_seq_len_
            );
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
    c10::optional<torch::Tensor> hidden_state_
) {
    std::cout << "Forwarding ShardTransformerDecoderImpl" << std::endl;
    torch::Tensor h;
    if (hidden_state_.has_value()) {
        h = hidden_state_.value();
    } else {
        h = tok_embeddings->forward(tokens_).to(model_dtype);
    }

    std::cout << "After token embeddings: " << h.sizes() << std::endl;

    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Processing layer " << i << std::endl;
        h = layers[i]->forward(
            h,
            mask_.value_or(torch::Tensor()),
            input_pos_.value_or(torch::Tensor())
        ).to(model_dtype);
    }

    if (shard.is_last_layer()) {
        h = norm->forward(h);
        auto out = output->forward(h);
        return out;
    } else {
        return h;
    }
}