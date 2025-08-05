// Read and load model configurations from config.json file
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

class ModelConfig {
    public:
        ModelConfig(
            const std::string config_path_
        );

        void load_config();

        const std::string config_path;
        float rope_scaling;
        int embed_dim;
        int num_heads;
        int head_dim;
        int num_kv_heads;
        int max_seq_len;
        int intermediate_size;
        float attn_dropout;
        int norm_eps;
        float rope_base; 
        int vocab_size;
        int num_layers;
        int attn_bias;
        std::string hidden_act;
        torch::ScalarType torch_dtype;
        bool use_cache;
        int original_max_seq_len = 1024;
};

#endif