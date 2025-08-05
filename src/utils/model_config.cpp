#include "model_config.h"
#include "helpers.h"

ModelConfig::ModelConfig(
  const std::string config_path_
) : config_path(config_path_){
  load_config();
}

void ModelConfig::load_config(){
  // load helpers
  Helpers model_helpers = Helpers();

  // Load the configuration from a JSON file
  std::ifstream config_file(config_path);
  if (!config_file.is_open()){
    throw std::runtime_error("Could not open config file: " + config_path);
  }

  nlohmann::json config_json = nlohmann::json::parse(config_file);
  config_file.close();

  // Parse the JSON configuration
  if(config_json.contains("rope_scaling") &&
     config_json["rope_scaling"].is_object()) {
    rope_scaling = config_json["rope_scaling"].value("factor", 32.0f);
    original_max_seq_len = config_json["rope_scaling"].value(
      "original_max_position_embeddings", 1024);
  } else if (config_json.contains("rope_scaling")) {
    rope_scaling = config_json["rope_scaling"].get<float>();
  } else {
    rope_scaling = 32.0f;
  }

  embed_dim = config_json.value("hidden_size", 256);
  num_heads = config_json.value("num_attention_heads", 32);
  head_dim = config_json.value("head_dim", embed_dim / num_heads);
  num_kv_heads = config_json.value("num_key_value_heads", 8);

  // need to shrink max_seq_len in low ram environements
  // update this to scale with how much ram is detected
  char *max_seq_len_env = std::getenv("CHEETAH_MAX_SEQ_LEN");
  if (max_seq_len_env != NULL){
    max_seq_len = std::atoi(max_seq_len_env);
  } else {
    if (model_helpers.is_low_memory("linux", 20000)){
      max_seq_len = original_max_seq_len;
    } else {
      max_seq_len = config_json.value("max_position_embeddings", 1024);
    }
  }

  intermediate_size = config_json.value("intermediate_size", 1024);
  attn_dropout = config_json.value("attention_dropout", 0.0f);
  norm_eps = config_json.value("rms_norm_eps", 1e-6f);
  rope_base = config_json.value("rope_theta", 500000.0f);
  vocab_size = config_json.value("vocab_size", 30522);
  num_layers = config_json.value("num_hidden_layers", 12);
  attn_bias = config_json.value("attention_bias", 0);
  hidden_act = config_json.value("hidden_act", "silu");
  use_cache = config_json.value("use_cache", true);

  std::string torch_dtype_name = config_json.value("torch_dtype", "bfloat16");
  if (torch_dtype_name == "bfloat16")
    torch_dtype = torch::kBFloat16;
  else if (torch_dtype_name == "float16")
    torch_dtype = torch::kFloat16;
  else if (torch_dtype_name == "float32")
    torch_dtype = torch::kFloat32;
  else if (torch_dtype_name == "int16")
    torch_dtype = torch::kInt16;
  else if (torch_dtype_name == "int32")
    torch_dtype = torch::kInt32;
  else
    throw std::runtime_error("Unsupported dtype: " + torch_dtype_name);
}
