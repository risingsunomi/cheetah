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

class TransformerSelfAttentionLayerImpl : public torch::nn::Module
{
public:
  TransformerSelfAttentionLayerImpl(
    int layer_idx_,
    MultiHeadAttention &attn_,
    MLP &mlp_,
    c10::optional<torch::nn::AnyModule> input_layernorm_,
    c10::optional<torch::nn::AnyModule> post_attn_layernorm_,
    c10::optional<torch::nn::AnyModule> input_scale_,
    c10::optional<torch::nn::AnyModule> post_attn_scale_,
    const c10::ScalarType &model_dtype_);

  torch::Tensor forward(
    const torch::Tensor &x_,
    const c10::optional<torch::Tensor> mask_,
    const c10::optional<torch::Tensor> input_pos_);

  void setup_cache(
    int &batch_size_,
    int &decoder_max_seq_len_);

  bool caches_are_setup() const;
  bool caches_are_enabled() const;
  void reset_cache();

private:
  int layer_idx = 0;
  MultiHeadAttention attn;
  MLP mlp;
  torch::nn::AnyModule input_layernorm;
  torch::nn::AnyModule post_attn_layernorm;
  torch::nn::AnyModule input_scale;
  torch::nn::AnyModule post_attn_scale;
  const c10::ScalarType model_dtype;
};

TORCH_MODULE(TransformerSelfAttentionLayer);

// ---- Sharded Transformer Decoder --- //
class ShardTransformerDecoderImpl : public torch::nn::Module
{
public:
  ShardTransformerDecoderImpl(
    const Shard &shard_,
    torch::nn::Embedding tok_embeddings_,
    std::vector<TransformerSelfAttentionLayer> layers_,
    int max_seq_len_,
    RMSNorm norm_,
    torch::nn::Linear output_,
    const c10::ScalarType &dtype_
  );

  void setup_caches(
    int batch_size_,
    int decoder_max_seq_len_);

  bool caches_are_enabled() const;
  void reset_caches();

  torch::Tensor forward(
    const torch::Tensor &tokens_,
    c10::optional<torch::Tensor> mask_,
    c10::optional<torch::Tensor> input_pos_,
    c10::optional<torch::Tensor> hidden_state_);

private:
  Shard shard;
  torch::nn::Embedding tok_embeddings{nullptr};
  std::vector<TransformerSelfAttentionLayer> layers;
  RMSNorm norm = nullptr;
  torch::nn::Linear output{nullptr};
  const c10::ScalarType model_dtype;
};
TORCH_MODULE(ShardTransformerDecoder);

#endif // TRANSFORMER_H