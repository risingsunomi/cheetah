#include <cmath>
#include "attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
  int64_t embed_dim,
  int64_t num_heads,
  int64_t num_kv_heads,
  int64_t head_dim,
  torch::nn::Linear q_proj,
  torch::nn::Linear k_proj,
  torch::nn::Linear v_proj,
  torch::nn::Linear out_proj,
  std::shared_ptr<RotaryEmbedding> pos_emb,
  std::shared_ptr<KVCache> kv_cache,
  bool is_causal,
  double attn_dropout,
  bool is_cache_enabled
) :
  embed_dim(embed_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  head_dim(head_dim),
  q_proj(q_proj),
  k_proj(k_proj),
  v_proj(v_proj),
  out_proj(out_proj),
  pos_emb(pos_emb),
  kv_cache(kv_cache),
  is_causal(is_causal),
  attn_dropout(attn_dropout),
  cache_enabled(is_cache_enabled) {
    register_module("q_proj", q_proj);
    register_module("k_proj", k_proj);
    register_module("v_proj", v_proj);
    register_module("out_proj", out_proj);
}

void MultiHeadAttentionImpl::setup_cache(
  int64_t batch_size,
  torch::Dtype dtype,
  int64_t encoder_max_cache_seq_len,
  int64_t decoder_max_cache_seq_len) {
  if (kv_cache) return;
  kv_cache = std::make_shared<KVCache>(
    batch_size, max_seq_len, num_kv_heads, head_dim, dtype);
  cache_enabled = true;
}

void MultiHeadAttentionImpl::reset_cache() {
  if (!kv_cache) {
    throw std::runtime_error("KV cache not initialized");
  }
  kv_cache->reset();
}

torch::Tensor MultiHeadAttentionImpl::forward(
  torch::Tensor x,
  torch::optional<torch::Tensor> y,
  torch::optional<torch::Tensor> mask,
  torch::optional<torch::Tensor> input_pos
) {
  auto B = x.size(0);
  auto sx = x.size(1);

  auto q = q_proj->forward(x);
  int64_t q_per_kv = num_heads / num_kv_heads;
  q = q.view({B, sx, num_kv_heads * q_per_kv, head_dim});

  if (pos_emb) {
    q = pos_emb->apply(q, input_pos.value_or(torch::Tensor()));
  }

  q = q.transpose(1, 2);

  if (y.has_value()) {
    auto y_tensor = y.value();
    auto sy = y_tensor.size(1);
    auto k = k_proj->forward(y_tensor).view({B, sy, num_kv_heads, head_dim});
    auto v = v_proj->forward(y_tensor).view({B, sy, num_kv_heads, head_dim});

    if (pos_emb) {
      k = pos_emb->apply(k, input_pos.value_or(torch::Tensor()));
    }

    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    std::tie(k, v) = kv_cache && cache_enabled ? kv_cache->update(k, v) : std::make_tuple(k, v);

    if (num_heads != num_kv_heads) {
      auto expanded_shape = std::vector<int64_t>{B, num_kv_heads, q_per_kv, -1, head_dim};
      k = k.unsqueeze(2).expand(expanded_shape).reshape({B, num_heads, -1, head_dim});
      v = v.unsqueeze(2).expand(expanded_shape).reshape({B, num_heads, -1, head_dim});
    }

    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
    if (mask.has_value()) {
      scores = scores.masked_fill(mask.value() == 0, -1e9);
    }
    auto attn = torch::softmax(scores, -1);
    auto out = torch::matmul(attn, v);
    out = out.transpose(1, 2).contiguous().view({B, sx, num_heads * head_dim});
    return out_proj->forward(out);
  }

  if (!kv_cache || !cache_enabled) {
    throw std::runtime_error("kv_cache is null and cache not enabled");
  }

  auto k = kv_cache->k_cache;
  auto v = kv_cache->v_cache;

  auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
  if (mask.has_value()) {
    scores = scores.masked_fill(mask.value() == 0, -1e9);
  }
  auto attn = torch::softmax(scores, -1);
  auto out = torch::matmul(attn, v);
  out = out.transpose(1, 2).contiguous().view({B, sx, num_heads * head_dim});
  return out_proj->forward(out);
}