#include <cmath>
#include "attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
  int embed_dim,
  int num_heads,
  int num_kv_heads,
  int head_dim,
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
  is_cache_enabled(is_cache_enabled) {
    register_module("q_proj", q_proj);
    register_module("k_proj", k_proj);
    register_module("v_proj", v_proj);
    register_module("out_proj", out_proj);
}

void MultiHeadAttentionImpl::setup_cache(
  int batch_size,
  torch::Dtype dtype,
  int max_seq_len) {
  
    if (kv_cache) return;
  
    kv_cache = std::make_shared<KVCache>(
      batch_size,
      max_seq_len,
      num_kv_heads,
      head_dim,
      dtype
    );
    
    is_cache_enabled = true;
}

void MultiHeadAttentionImpl::reset_cache() {
  if (!kv_cache) {
    throw std::runtime_error("KV cache not initialized");
  }
  kv_cache->reset();
}

torch::Tensor MultiHeadAttentionImpl::forward(
    torch::Tensor query_input,
    c10::optional<torch::Tensor> key_value_input,
    c10::optional<torch::Tensor> attention_mask,
    c10::optional<torch::Tensor> input_positions
) {
    const int batch_size = query_input.size(0);
    const int query_seq_len = query_input.size(1);

    torch::Tensor query_proj = q_proj->forward(query_input);
    const int queries_per_kv_head = num_heads / num_kv_heads;
    query_proj = query_proj.view({batch_size, query_seq_len, num_kv_heads * queries_per_kv_head, head_dim});

    if (pos_emb) {
        query_proj = pos_emb->apply(query_proj, input_positions.value_or(torch::Tensor()));
    }

    query_proj = query_proj.transpose(1, 2);
    if (key_value_input.has_value()) {
        const torch::Tensor& kv_input = key_value_input.value();
        const int kv_seq_len = kv_input.size(1);
        torch::Tensor key_proj = k_proj->forward(kv_input).view({batch_size, kv_seq_len, num_kv_heads, head_dim});
        torch::Tensor value_proj = v_proj->forward(kv_input).view({batch_size, kv_seq_len, num_kv_heads, head_dim});

        if (pos_emb) {
            key_proj = pos_emb->apply(key_proj, input_positions.value_or(torch::Tensor()));
        }

        key_proj = key_proj.transpose(1, 2);
        value_proj = value_proj.transpose(1, 2);


        if (kv_cache && is_cache_enabled) {
            std::tie(key_proj, value_proj) = kv_cache->update(key_proj, value_proj);
        }

        if (num_heads != num_kv_heads) {
            std::vector<int> expanded_shape = {
                batch_size, num_kv_heads, queries_per_kv_head, -1, head_dim
            };
            
            key_proj = key_proj.unsqueeze(2)
              .expand(expanded_shape)
              .reshape({batch_size, num_heads, -1, head_dim});
            
            value_proj = value_proj.unsqueeze(2)
              .expand(expanded_shape)
              .reshape({batch_size, num_heads, -1, head_dim});
        }

        torch::Tensor attn_scores = torch::matmul(query_proj, key_proj.transpose(-2, -1));
        attn_scores /= std::sqrt(static_cast<double>(head_dim));
        if (attention_mask.has_value()) {
            attn_scores = attn_scores.masked_fill(attention_mask.value() == 0, -1e9);
        }
        torch::Tensor attn_weights = torch::softmax(attn_scores, -1);
        torch::Tensor attn_output = torch::matmul(attn_weights, value_proj);

        attn_output = attn_output.transpose(1, 2)
          .contiguous()
          .view({batch_size, query_seq_len, num_heads * head_dim});

        return out_proj->forward(attn_output);
    }

    if (!kv_cache || !is_cache_enabled) {
        throw std::runtime_error("KV cache is not initialized and no key/value input was provided.");
    }

    torch::Tensor key_proj = kv_cache->k_cache;
    torch::Tensor value_proj = kv_cache->v_cache;

    torch::Tensor attn_scores = torch::matmul(query_proj, key_proj.transpose(-2, -1));
    attn_scores /= std::sqrt(static_cast<double>(head_dim));

    if (attention_mask.has_value()) {
        attn_scores = attn_scores.masked_fill(attention_mask.value() == 0, -1e9);
    }

    torch::Tensor attn_weights = torch::softmax(attn_scores, -1);
    torch::Tensor attn_output = torch::matmul(attn_weights, value_proj);

    attn_output = attn_output.transpose(1, 2)
      .contiguous()
      .view({batch_size, query_seq_len, num_heads * head_dim});

    return out_proj->forward(attn_output);
}
