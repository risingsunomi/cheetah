#include "general_mha_model.h"

GeneralMHAModel::GeneralMHAModel(
  const Shard &shard_,
  const ModelConfig &config_,
  const std::string safetensors_path_,
  bool &use_cache_
) : shard(shard_),
  config(config_),
  use_cache(use_cache_),
  safetensors_path(safetensors_path_) {
  // load helpers
  Helpers model_helpers = Helpers();

  // load safetensors
  std::cout << "generlmha constructor called" << std::endl;
  std::cout << "Loading safetensors from " + safetensors_path_ << std::endl;
  SafeTensorsLoader safetensors_loader(safetensors_path_);
  auto model_weights = safetensors_loader.getTensors();

  // will make post and prefixes customizable but for now hard coded
  // assuming llama weight naming convention but will add better detection
  std::string model_prefix = "model.";
  std::string model_layer_prefix = "model.layers.";
  std::string model_postfix = ".weight";

  // Create decoder layers from layer_start to layer_end
  for (int i = shard.start_layer; i < shard.end_layer; ++i)
  {
    // --- weight and component loading
    torch::nn::Linear q_proj(config.embed_dim, config.embed_dim);
    std::string q_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".self_attn.q_proj" + model_postfix;
    torch::Tensor q_proj_weight = model_weights.at(q_proj_weight_name);
    q_proj->weight = q_proj_weight;

    torch::nn::Linear k_proj(config.embed_dim, config.embed_dim);
    std::string k_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".self_attn.k_proj" + model_postfix;
    torch::Tensor k_proj_weight = model_weights.at(k_proj_weight_name);
    k_proj->weight = k_proj_weight;

    torch::nn::Linear v_proj(config.embed_dim, config.embed_dim);
    std::string v_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".self_attn.v_proj" + model_postfix;
    torch::Tensor v_proj_weight = model_weights.at(v_proj_weight_name);
    v_proj->weight = v_proj_weight;

    torch::nn::Linear o_proj(config.embed_dim, config.embed_dim);
    std::string o_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".self_attn.o_proj" + model_postfix;
    torch::Tensor o_proj_weight = model_weights.at(o_proj_weight_name);
    o_proj->weight = o_proj_weight;

    MLP mlp(config.embed_dim,
      config.intermediate_size,
      config.hidden_act);
    
    std::string up_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".mlp.up_proj" + model_postfix;
    torch::Tensor up_proj_weight = model_weights.at(up_proj_weight_name);
    mlp->up_proj->weight = up_proj_weight;

    std::string down_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".mlp.down_proj" + model_postfix;
    torch::Tensor down_proj_weight = model_weights.at(down_proj_weight_name);
    mlp->down_proj->weight = down_proj_weight;

    std::string gate_proj_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".mlp.gate_proj" + model_postfix;
    torch::Tensor gate_proj_weight = model_weights.at(gate_proj_weight_name);
    mlp->gate_proj->weight = gate_proj_weight;

    std::string post_attn_layernorm_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".post_attention_layernorm" + model_postfix;
    auto post_attn_layernorm_weight = model_weights.at(post_attn_layernorm_weight_name);
    auto post_attn_layernorm = RMSNorm(config.embed_dim);
    post_attn_layernorm->weight = post_attn_layernorm_weight;

    std::string input_layernorm_weight_name = model_layer_prefix +
      std::to_string(i) +
      ".input_layernorm" + model_postfix;
    auto input_layernorm_weight = model_weights.at(input_layernorm_weight_name);
    auto input_layernorm = RMSNorm(config.embed_dim);
    input_layernorm->weight = input_layernorm_weight;

    // ---

    auto mha = MultiHeadAttention(
      i,
      config.embed_dim,
      config.num_heads,
      config.num_kv_heads,
      config.head_dim,
      q_proj,
      k_proj,
      v_proj,
      o_proj,
      RotaryEmbedding(
        config.head_dim,
        config.max_seq_len,
        config.rope_base),
      c10::nullopt,
      true,
      config.attn_dropout,
      use_cache,
      config.torch_dtype
    );

    auto transformer_layer = TransformerSelfAttentionLayer(
      i,
      mha,
      mlp,
      torch::nn::AnyModule(post_attn_layernorm),
      torch::nn::AnyModule(input_layernorm),
      c10::nullopt,
      c10::nullopt,
      config.torch_dtype
    );

    self_attn_layers.push_back(transformer_layer);
  }

  // --- weight and component loading
  auto embed_weight = model_weights.at(
      model_prefix + "embed_tokens" + model_postfix);
  auto tok_embedding = torch::nn::Embedding(config.vocab_size, config.embed_dim);
  tok_embedding->weight = embed_weight;

  auto std_out_proj = torch::nn::Linear(config.embed_dim, config.vocab_size);

  bool use_tied = false;
  if (model_helpers.llama_detect(shard.model_id)){
    // if 3.2 tie or use same weight as embeddings
    if (model_helpers.model_ver_detect(shard.model_id, "3.2")){
      std_out_proj->weight = embed_weight;
      use_tied = true;
    } 
  }

  if(!use_tied) {
    // use lm_head
    auto lm_head_weight = model_weights.at(
        "lm_head" + model_postfix);
    std_out_proj->weight = lm_head_weight;
  }

  auto rms_norm_weight = model_weights.at(
      model_prefix + "norm" + model_postfix);
  auto rms_norm = RMSNorm(config.embed_dim);
  rms_norm->weight = rms_norm_weight;

  // ---

  shard_decoder = ShardTransformerDecoder(
      shard,
      tok_embedding,
      self_attn_layers,
      config.max_seq_len,
      rms_norm,
      std_out_proj,
      config.torch_dtype
    );
}

torch::Tensor GeneralMHAModel::forward(
    const torch::Tensor &tokens_,
    const c10::optional<torch::Tensor> mask_,
    const c10::optional<torch::Tensor> input_pos_,
    const c10::optional<torch::Tensor> hidden_state_
){
  if (use_cache) {
    shard_decoder->setup_caches(
        1,
        config.max_seq_len
      );
  }
  return shard_decoder->forward(tokens_, mask_, input_pos_, hidden_state_);
}
