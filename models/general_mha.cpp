#include "general_mha.h"
#include <cmath>

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
