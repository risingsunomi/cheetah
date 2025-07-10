#include <torch/torch.h>
#include <iostream>
#include "transformer/general_mha_model.h"
#include "utils/shard.h"

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
  auto shard = Shard("test_model", layer_start, layer_end, layer_total);
  auto model = GeneralMHAModel(
    shard,
    vocab_size,
    embed_dim,
    hidden_dim,
    num_heads,
    num_kv_heads,
    head_dim,
    seq_len,
    true
  );

  // * currently testing only self attention layer *

  model.eval();

  // === Inputs ===
  auto tokens = torch::randint(0, vocab_size, {1, seq_len}, torch::kLong);
  auto mask = torch::ones({seq_len, seq_len}, torch::kBool).tril(); 
  mask = mask.unsqueeze(0);
  auto input_pos = torch::arange(seq_len, torch::kInt32).unsqueeze(0);

  // === Forward ===
  auto output = model.forward(
    tokens,
    mask,
    input_pos,
    c10::nullopt
  );

  // === Output ===
  std::cout << "Self Attention Layer Output shape: " << output.sizes() << std::endl;
  std::cout << "First token logits (first 10 values):" << std::endl;
  std::cout << output[0][0].slice(0, 0, 10) << std::endl;

  return 0;
}
