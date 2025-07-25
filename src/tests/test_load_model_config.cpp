#include "./../utils/shard.h"
#include "./../utils/model_config.h"
#include "./../general_mha_model.h"


#include <iostream>

int main() {
    try {
        const std::string config_path("/home/t0kenl1mit/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/eb49081324edb2ff14f848ce16393c067c6f4976/config.json");
        std::cout << "Loading model configuration from: " << config_path << std::endl;
        
        ModelConfig config(config_path);
        std::cout << "Model configuration loaded successfully." << std::endl;
        std::cout << "Rope Scaling: " << config.rope_scaling << std::endl;
        std::cout << "Embed Dim: " << config.embed_dim << std::endl;
        std::cout << "Num Heads: " << config.num_heads << std::endl;
        std::cout << "Head Dim: " << config.head_dim << std::endl;
        std::cout << "Num KV Heads: " << config.num_kv_heads << std::endl;
        std::cout << "Max Seq Len: " << config.max_seq_len << std::endl;

        std::cout << "Loading General MHA..." << std::endl;

        const std::string model_name("Llama-3.2-1B-Instruct");
        Shard shard(
            model_name,
            0,
            21,
            22
        );

        auto general_mha = GeneralMHAModel(
            shard,
            config,
            config.use_cache
        );
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model configuration: " << e.what() << std::endl;
    }


    return 0;
}