#include <torch/torch.h>
#include <iostream>
#include <string>
#include "src/general_mha_model.h"
#include "src/utils/shard.h"
#include "src/utils/model_config.h"

int main() {
  torch::manual_seed(42);

  // Model hyperparameters
  int layer_start = 0;
  int layer_end = 1;
  int layer_total = 2;

  const std::string config_path(
    "/home/t0kenl1mit/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/eb49081324edb2ff14f848ce16393c067c6f4976/config.json");
  ModelConfig config(config_path);

  // Initialize model
  auto shard = Shard("test_model", layer_start, layer_end, layer_total);
  auto model = GeneralMHAModel(
    shard,
    config,
    config.use_cache
  );

  // * currently testing only self attention layer *

  model.eval();

  // === Inputs ===
  auto tokens = torch::randint(0, config.vocab_size, {1, config.max_seq_len}, torch::kLong);  // [batch, seq_len]
  auto mask = torch::ones({config.max_seq_len, config.max_seq_len}, torch::kBool).tril();  // [seq_len, seq_len]
  mask = mask.unsqueeze(0);  // [1, seq_len, seq_len] — batch dim added
  auto input_pos = torch::arange(0, config.max_seq_len, torch::kInt32).unsqueeze(0);  // [1, seq_len]

  std::cout << "Input tokens shape: " << tokens.sizes() << std::endl;
  std::cout << "Input mask shape: " << mask.sizes() << std::endl;
  std::cout << "Input positions shape: " << input_pos.sizes() << std::endl;

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
