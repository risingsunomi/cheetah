#include "general_mha.h"
#include <cmath>

// RMSNorm
RMSNormImpl::RMSNormImpl(int64_t hidden_size, float eps)
    : eps(eps) {
  weight = register_parameter("weight", torch::ones({hidden_size}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& input) {
  auto norm_x = input.norm(2, -1, true);
  return (input / (norm_x / std::sqrt(input.size(-1)) + eps)) * weight;
}

// MLP
MLPImpl::MLPImpl(int64_t input_dim, int64_t hidden_dim) {
  gate_proj = register_module("gate_proj", torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_dim).bias(false)));
  up_proj = register_module("up_proj", torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_dim).bias(false)));
  down_proj = register_module("down_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, input_dim).bias(false)));
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  auto gate = torch::silu(gate_proj->forward(x));
  auto up = up_proj->forward(x);
  return down_proj->forward(gate * up);
}

// MultiHeadAttention
MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t embed_dim, int64_t num_heads, int64_t head_dim, int64_t max_seq_len)
    : num_heads(num_heads), head_dim(head_dim), rope(head_dim, max_seq_len) {
  q_proj = register_module("q_proj", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, num_heads * head_dim).bias(true)));
  k_proj = register_module("k_proj", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, num_heads * head_dim).bias(true)));
  v_proj = register_module("v_proj", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, num_heads * head_dim).bias(true)));
  out_proj = register_module("out_proj", torch::nn::Linear(torch::nn::LinearOptions(num_heads * head_dim, embed_dim).bias(true)));
}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x, torch::Tensor mask) {
  auto B = x.size(0), T = x.size(1);
  auto q = q_proj->forward(x).view({B, T, num_heads, head_dim}).transpose(1, 2);
  auto k = k_proj->forward(x).view({B, T, num_heads, head_dim}).transpose(1, 2);
  auto v = v_proj->forward(x).view({B, T, num_heads, head_dim}).transpose(1, 2);

  q = rope.apply_rotary(q);
  k = rope.apply_rotary(k);

  auto attn_scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt((double)head_dim);
  if (mask.defined()) {
    attn_scores = attn_scores.masked_fill(mask == 0, -1e9);
  }

  auto attn_weights = torch::softmax(attn_scores, -1);
  auto attn_output = torch::matmul(attn_weights, v);
  attn_output = attn_output.transpose(1, 2).contiguous().view({B, T, num_heads * head_dim});
  return out_proj->forward(attn_output);
}

// TransformerBlock
TransformerBlockImpl::TransformerBlockImpl(int64_t embed_dim, int64_t hidden_dim, int64_t num_heads, int64_t head_dim, int64_t max_seq_len)
    : mha(register_module("mha", MultiHeadAttention(embed_dim, num_heads, head_dim, max_seq_len))),
      mlp(register_module("mlp", MLP(embed_dim, hidden_dim))),
      norm1(register_module("norm1", RMSNorm(embed_dim))),
      norm2(register_module("norm2", RMSNorm(embed_dim))) {}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x, torch::Tensor mask) {
  x = x + mha->forward(norm1->forward(x), mask);
  x = x + mlp->forward(norm2->forward(x));
  return x;
}

// GeneralMHA
GeneralMHAImpl::GeneralMHAImpl(int64_t vocab_size, int64_t embed_dim, int64_t hidden_dim,
                               int64_t num_heads, int64_t head_dim, int64_t max_seq_len)
    : tok_embeddings(register_module("tok_embeddings", torch::nn::Embedding(vocab_size, embed_dim))),
      output_proj(register_module("output_proj", torch::nn::Linear(torch::nn::LinearOptions(embed_dim, vocab_size).bias(false)))),
      block(register_module("block", TransformerBlock(embed_dim, hidden_dim, num_heads, head_dim, max_seq_len))),
      final_norm(register_module("final_norm", RMSNorm(embed_dim))) {}

torch::Tensor GeneralMHAImpl::forward(torch::Tensor tokens, torch::Tensor mask) {
  auto x = tok_embeddings->forward(tokens);
  x = block->forward(x, mask);
  x = final_norm->forward(x);
  return output_proj->forward(x);
}
