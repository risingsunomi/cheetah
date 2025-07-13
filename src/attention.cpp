#include <cmath>
#include "attention.h"

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
  int& embed_dim_,
  int& num_heads_,
  int& num_kv_heads_,
  int& head_dim_,
  torch::nn::Linear q_proj_,
  torch::nn::Linear k_proj_,
  torch::nn::Linear v_proj_,
  torch::nn::Linear out_proj_,
  std::shared_ptr<RotaryEmbedding> pos_emb_,
  std::shared_ptr<KVCache> kv_cache_,
  bool& is_causal_,
  float& attn_dropout_,
  bool& use_cache_
) :
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
  int& batch_size_,
  torch::Dtype& dtype_,
  int& max_seq_len_) {
  
    if (&kv_cache != nullptr) return;
  
    kv_cache = std::make_shared<KVCache>(
      batch_size_,
      max_seq_len_,
      num_kv_heads,
      head_dim,
      dtype_
    );
    
    use_cache = true;
}

void MultiHeadAttentionImpl::reset_cache() {
  if (!kv_cache) {
    throw std::runtime_error("KV cache not initialized");
  }
  kv_cache->reset();
}

torch::Tensor MultiHeadAttentionImpl::forward(
    torch::Tensor& query_input_,
    c10::optional<torch::Tensor&> key_value_input_,
    c10::optional<torch::Tensor&> attention_mask_,
    c10::optional<torch::Tensor&> input_positions_
) {
    const int batch_size = query_input_.size(0);
    const int query_seq_len = query_input_.size(1);

    torch::Tensor query_proj = q_proj->forward(query_input_);
    const int queries_per_kv_head = num_heads / num_kv_heads;
    query_proj = query_proj.view({batch_size, query_seq_len, num_kv_heads * queries_per_kv_head, head_dim});

    if (pos_emb) {
        query_proj = pos_emb->apply(query_proj, input_positions_.value_or(torch::Tensor()));
    }

    query_proj = query_proj.transpose(1, 2);
    if (key_value_input_) {
        const torch::Tensor& kv_input = key_value_input_.value();
        const int kv_seq_len = kv_input.size(1);
        torch::Tensor key_proj = k_proj->forward(kv_input).view({batch_size, kv_seq_len, num_kv_heads, head_dim});
        torch::Tensor value_proj = v_proj->forward(kv_input).view({batch_size, kv_seq_len, num_kv_heads, head_dim});

        if (pos_emb) {
            key_proj = pos_emb->apply(key_proj, input_positions_.value_or(torch::Tensor()));
        }

        key_proj = key_proj.transpose(1, 2);
        value_proj = value_proj.transpose(1, 2);


        if (kv_cache && use_cache) {
            std::tie(key_proj, value_proj) = kv_cache->update(key_proj, value_proj);
        }

        if (num_heads != num_kv_heads) {
            key_proj = key_proj.unsqueeze(2)
              .expand({
                batch_size, num_kv_heads, queries_per_kv_head, -1, head_dim})
              .reshape({batch_size, num_heads, -1, head_dim});
            
            value_proj = value_proj.unsqueeze(2)
              .expand({
                batch_size, num_kv_heads, queries_per_kv_head, -1, head_dim})
              .reshape({batch_size, num_heads, -1, head_dim});
        }

        torch::Tensor attn_scores = torch::matmul(query_proj, key_proj.transpose(-2, -1));
        attn_scores /= std::sqrt(static_cast<double>(head_dim));
        if (attention_mask_) {
            attn_scores = attn_scores.masked_fill(attention_mask_.value() == 0, -1e9);
        }
        torch::Tensor attn_weights = torch::softmax(attn_scores, -1);
        torch::Tensor attn_output = torch::matmul(attn_weights, value_proj);

        attn_output = attn_output.transpose(1, 2)
          .contiguous()
          .view({batch_size, query_seq_len, num_heads * head_dim});

        return out_proj->forward(attn_output);
    }

    if (!kv_cache || !use_cache) {
        throw std::runtime_error("KV cache is not initialized and no key/value input was provided.");
    }

    torch::Tensor key_proj = kv_cache->k_cache;
    torch::Tensor value_proj = kv_cache->v_cache;

    torch::Tensor attn_scores = torch::matmul(query_proj, key_proj.transpose(-2, -1));
    attn_scores /= std::sqrt(static_cast<double>(head_dim));

    if (attention_mask_) {
        attn_scores = attn_scores.masked_fill(attention_mask_.value() == 0, -1e9);
    }

    torch::Tensor attn_weights = torch::softmax(attn_scores, -1);
    torch::Tensor attn_output = torch::matmul(attn_weights, value_proj);

    attn_output = attn_output.transpose(1, 2)
      .contiguous()
      .view({batch_size, query_seq_len, num_heads * head_dim});

    return out_proj->forward(attn_output);
}
