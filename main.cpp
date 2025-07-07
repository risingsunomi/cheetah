#include <torch/torch.h>
#include <iostream>
#include "models/general_model.h"

int main() {
  torch::manual_seed(42);

  // Model hyperparameters
  int64_t layer_start = 0;
  int64_t layer_end = 1;
  int64_t layer_total = 1;
  int64_t vocab_size = 30522;
  int64_t embed_dim = 256;
  int64_t hidden_dim = 1024;
  int64_t num_heads = 8;
  int64_t num_kv_heads = 8;
  int64_t head_dim = 32;
  int64_t seq_len = 10;

  // Initialize model
  auto model = GeneralMHAModel(
    layer_start,
    layer_end,
    layer_total,
    vocab_size,
    embed_dim,
    hidden_dim,
    num_heads,
    num_kv_heads,
    head_dim,
    seq_len
  );

  model->eval();

  // Random token input
  auto tokens = torch::randint(0, vocab_size, {1, seq_len}, torch::kInt64);

  // Run forward pass
  auto output = model->forward(tokens);

  // Print outputs
  std::cout << "Output shape: " << output.sizes() << std::endl;
  std::cout << "First token logits (first 10 values):" << std::endl;
  std::cout << output[0][0].slice(0, 0, 10) << std::endl;

  return 0;
}
