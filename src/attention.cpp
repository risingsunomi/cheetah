#include <cmath>
#include <iostream>
#include "attention.h"

using namespace torch;

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
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
  bool use_cache_
) :
  layer_id(layer_id_),
  embed_dim(embed_dim_),
  num_heads(num_heads_),
  num_kv_heads(num_kv_heads_),
  head_dim(head_dim_),
  q_proj(q_proj_),
  k_proj(k_proj_),
  v_proj(v_proj_),
  out_proj(out_proj_),
  pos_emb(pos_emb_),
  kv_cache(kv_cache_),
  is_causal(is_causal_),
  attn_dropout(attn_dropout_),
  use_cache(use_cache_) {
    register_module("q_proj", q_proj);
    register_module("k_proj", k_proj);
    register_module("v_proj", v_proj);
    register_module("out_proj", out_proj);
}

void MultiHeadAttentionImpl::setup_cache(
  int batch_size_,
  torch::Dtype dtype_,
  int max_seq_len_) {

    std::cout << "MHA Setting up Cache for layer " << layer_id << std::endl;
  
    if (kv_cache) return;
  
    kv_cache = KVCache(
      batch_size_,
      max_seq_len_,
      num_kv_heads,
      head_dim,
      dtype_
    );
    
    use_cache = true;
    cache_enabled = true;
}

void MultiHeadAttentionImpl::reset_cache() {
  if (cache_enabled) {
    kv_cache->reset();
  }
  
}

Tensor MultiHeadAttentionImpl::forward(
  Tensor& query_input_,
  c10::optional<Tensor> key_value_input_,
  c10::optional<Tensor> attention_mask_,
  c10::optional<Tensor> input_positions_
) {
  const int B = query_input_.size(0);
  const int S_q = query_input_.size(1);
  const int q_per_kv = num_heads / num_kv_heads;

  // Project query
  Tensor q = q_proj->forward(query_input_);  // [B, S_q, num_heads * head_dim]
  q = q.view({B, S_q, num_kv_heads * q_per_kv, head_dim});
  q = pos_emb.forward(q, input_positions_.value_or(Tensor()));
  q = q.transpose(1, 2);  // [B, num_heads, S_q, head_dim]

  Tensor k, v;
  if (key_value_input_.has_value()) {
    Tensor kv = key_value_input_.value();
    const int S_kv = kv.size(1);

    k = k_proj->forward(kv).view({B, S_kv, num_kv_heads, head_dim});
    v = v_proj->forward(kv).view({B, S_kv, num_kv_heads, head_dim});
    k = pos_emb.forward(k, input_positions_.value_or(Tensor()));
    k = k.transpose(1, 2);  // [B, num_kv_heads, S_kv, head_dim]
    v = v.transpose(1, 2);

    if (kv_cache && cache_enabled) {
      std::tie(k, v) = kv_cache->update(k, v);
    }
  } else {
    TORCH_CHECK(kv_cache && cache_enabled, "KV cache must be enabled for decoding.");
    k = kv_cache->k_cache;
    v = kv_cache->v_cache;
  }

  // GQA: expand kv to match query heads if needed
  if (num_heads != num_kv_heads) {
    auto expand_shape = std::vector<int64_t>{B, num_kv_heads, q_per_kv, -1, head_dim};
    k = k.unsqueeze(2).expand(expand_shape).reshape({B, num_heads, k.size(2), head_dim});
    v = v.unsqueeze(2).expand(expand_shape).reshape({B, num_heads, v.size(2), head_dim});
  }

  // Scaled dot-product attention
  Tensor attn_scores = torch::matmul(q, k.transpose(-2, -1));
  attn_scores = attn_scores / std::sqrt((double)head_dim);

  if (attention_mask_) {
    attn_scores = attn_scores.masked_fill(attention_mask_.value() == 0, -1e9);
  }

  Tensor attn_weights = torch::softmax(attn_scores, -1);
  Tensor attn_output = torch::matmul(attn_weights, v);  // [B, num_heads, S_q, head_dim]

  attn_output = attn_output.transpose(1, 2).contiguous().view({B, S_q, num_heads * head_dim});
  return out_proj->forward(attn_output);
}