#include <torch/torch.h>
#include <iostream>
#include "src/general_mha_model.h"
#include "src/utils/shard.h"

int main() {
  torch::manual_seed(42);

  // Model hyperparameters
  int64_t layer_start = 0;
  int64_t layer_end = 21;
  int64_t layer_total = 22;
  float_t rope_scaling = 32.0; // rope_scaling.factor *
  int64_t vocab_size = 30522; // vocab_size *
  int64_t embed_dim = 256; // hidden_size *
  int64_t hidden_dim = 1024; // hidden_size / num_attention_heads *
  int64_t num_heads = 8; // *
  int64_t num_kv_heads = 8; // num_key_value_heads *
  int64_t head_dim = 32; // *
  int64_t seq_len = 10; // max_seq_len - max_position_embeddings
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

  std::cout << "Input tokens shape: " << tokens.sizes() << std::endl;
  std::cout << "Input mask shape: " << mask.sizes() << std::endl;
  std::cout << "Input positions shape: " << input_pos.sizes() << std::endl;
  std::cout << "Input tokens: " << tokens << std::endl;

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
