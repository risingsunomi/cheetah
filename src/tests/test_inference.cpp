#include <torch/torch.h>
#include <iostream>
#include <string>
#include "../general_mha_model.h"
#include "../utils/shard.h"
#include "../utils/model_config.h"

int main()
{
    torch::manual_seed(42);

    // Model hyperparameters
    int layer_start = 0;
    int layer_end = 1;
    int layer_total = 2;

    const std::string config_path(
        "/home/t0kenl1mit/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/eb49081324edb2ff14f848ce16393c067c6f4976/config.json");
    
    std::cout << "Loading model config @ " + config_path << std::endl;
    ModelConfig config(config_path);

    // Initialize model
    std::cout << "Loading shard information" << std::endl;
    std::cout << "layer_start " << std::to_string(layer_start) << std::endl;
    std::cout << "layer_end " << std::to_string(layer_end) << std::endl;
    std::cout << "layer_total " << std::to_string(layer_total) << std::endl;
    auto shard = Shard("Llama-3.2-1B-Instruct", layer_start, layer_end, layer_total);

    const std::string safetensors_path(
        "/home/t0kenl1mit/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/eb49081324edb2ff14f848ce16393c067c6f4976/model.safetensors");
    
    std::cout << "Loading model with weights @ " << safetensors_path << std::endl;
    auto model = GeneralMHAModel(
        shard,
        config,
        safetensors_path,
        config.use_cache);

    // * currently testing only self attention layer *

    model.eval();

    // === Inputs ===
    auto tokens = torch::randint(0, config.vocab_size, {1, config.max_seq_len}, torch::kLong); // [batch, seq_len]
    auto mask = torch::ones({config.max_seq_len, config.max_seq_len}, torch::kBool).tril();    // [seq_len, seq_len]
    mask = mask.unsqueeze(0);                                                                  // [1, seq_len, seq_len] — batch dim added
    auto input_pos = torch::arange(0, config.max_seq_len, torch::kInt32).unsqueeze(0);         // [1, seq_len]

    std::cout << "Input tokens shape: " << tokens.sizes() << std::endl;
    std::cout << "Input mask shape: " << mask.sizes() << std::endl;
    std::cout << "Input positions shape: " << input_pos.sizes() << std::endl;

    // === Forward ===
    auto output = model.forward(
        tokens,
        mask,
        input_pos,
        c10::nullopt);

    // === Output ===
    std::cout << "Self Attention Layer Output shape: " << output.sizes() << std::endl;
    std::cout << "First token logits (first 10 values):" << std::endl;
    std::cout << output[0][0].slice(0, 0, 10) << std::endl;
}