#include "model_config.h"
#include "system_utils.h"

ModelConfig::ModelConfig(
    const std::string config_path_
) : config_path(config_path_) {
    load_config();
}

void ModelConfig::load_config() {
    // Load the configuration from a JSON file
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }
    
    std::cout << "Parsing model configuration from: " << config_path << std::endl;
    nlohmann::json config_json = nlohmann::json::parse(config_file);
    std::cout << "Model configuration parsed successfully." << std::endl;
    std::cout << config_json << std::endl;
    config_file.close();

    // Parse the JSON configuration
    rope_scaling = config_json["rope_scaling"].value("factor", 32.0f);
    embed_dim = config_json.value("hidden_size", 256);
    num_heads = config_json.value("num_attention_heads", 32);
    head_dim = config_json.value("head_dim", embed_dim / num_heads);
    num_kv_heads = config_json.value("num_key_value_heads", 8);
    
    // need to shrink max_seq_len in low ram environements
    // update this to scale with how much ram is detected
    if(is_low_memory()) {
        max_seq_len = config_json["rope_scaling"].value(
            "original_max_position_embeddings", 8192);
        std::cout << 
            "LOW RAM DETECTED (>20GB), USING ORIGINAL MAX SEQ LEN " <<
            std::to_string(max_seq_len) << 
            std::endl;
    } else {
        max_seq_len = config_json.value("max_position_embeddings", 1024);
        std::cout << "MAX SEQ LEN SET TO " << std::to_string(max_seq_len) << std::endl;
    }
    

    intermediate_size = config_json.value("intermediate_size", 1024);
    attn_dropout = config_json.value("attention_dropout", 0.0f);
    norm_eps = config_json.value("rms_norm_eps", 1e-6f);
    rope_base = config_json.value("rope_theta", 500000.0f);
    vocab_size = config_json.value("vocab_size", 30522);
    num_layers = config_json.value("num_hidden_layers", 12);
    attn_bias = config_json.value("attention_bias", 0);
    hidden_act = config_json.value("hidden_act", "silu");
    torch_dtype = config_json.value("torch_dtype", "bfloat32");
    use_cache = config_json.value("use_cache", true);
}
