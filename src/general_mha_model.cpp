#include "general_mha_model.h"

GeneralMHAModel::GeneralMHAModel(
  const Shard shard_,
  const ModelConfig config_,
  const std::string model_path_,
  bool &use_cache_
) : shard(shard_),
  config(config_),
  model_path(model_path_),
  use_cache(use_cache_) {
  // load helpers
  Helpers model_helpers = Helpers();

  // safetensor loader
  SafeTensorsLoader st_loader(
    model_path,
    shard
  );

  // Create decoder layers from layer_start to layer_end
  for (int i = shard.start_layer; i < shard.end_layer; ++i)
  {
    // --- weight and component loading
    torch::nn::Linear q_proj(config.embed_dim, config.embed_dim);
    q_proj->weight = st_loader.findWeight(
      model_layer_prefix +
        std::to_string(i) +
        ".self_attn.q_proj" + model_postfix
    );

    torch::nn::Linear k_proj(config.embed_dim, config.embed_dim);
    k_proj->weight = st_loader.findWeight(
      model_layer_prefix +
        std::to_string(i) +
        ".self_attn.k_proj" + model_postfix
    );

    torch::nn::Linear v_proj(config.embed_dim, config.embed_dim);
    v_proj->weight = st_loader.findWeight(
      model_layer_prefix +
        std::to_string(i) +
        ".self_attn.v_proj" + model_postfix
    );

    torch::nn::Linear o_proj(config.embed_dim, config.embed_dim);
    o_proj->weight = st_loader.findWeight(
      model_layer_prefix +
      std::to_string(i) +
      ".self_attn.o_proj" + model_postfix
    );

    MLP mlp(config.embed_dim,
      config.intermediate_size,
      config.hidden_act);
    mlp->up_proj->weight = st_loader.findWeight(
      model_layer_prefix + 
      std::to_string(i) + 
      ".mlp.up_proj"
    );
    mlp->down_proj->weight = st_loader.findWeight(
      model_layer_prefix +
      std::to_string(i) +
      ".mlp.down_proj" + model_postfix
    );
    mlp->gate_proj->weight = st_loader.findWeight(
      model_layer_prefix +
      std::to_string(i) + ".mlp.gate_proj"
    );

    auto post_attn_layernorm = RMSNorm(config.embed_dim);
    post_attn_layernorm->weight = st_loader.findWeight(
      model_layer_prefix +
      std::to_string(i) +
      ".post_attention_layernorm" + model_postfix
    );

    auto input_layernorm = RMSNorm(config.embed_dim);
    input_layernorm->weight = st_loader.findWeight(
      model_layer_prefix +
      std::to_string(i) +
      ".input_layernorm" + model_postfix
    );

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
  auto tok_embedding = torch::nn::Embedding(config.vocab_size, config.embed_dim);
  tok_embedding->weight = st_loader.findWeight(
    model_prefix + "embed_tokens" + model_postfix
  );

  auto std_out_proj = torch::nn::Linear(config.embed_dim, config.vocab_size);

  bool use_tied = false;
  if (model_helpers.llama_detect(shard.model_id)){
    // if 3.2 tie or use same weight as embeddings
    if (model_helpers.model_ver_detect(shard.model_id, "3.2")){
      std_out_proj->weight = st_loader.findWeight(
        model_prefix + "embed_tokens" + model_postfix
      );
      use_tied = true;
    } 
  }

  if(!use_tied) {
    // use lm_head
    std_out_proj->weight = st_loader.findWeight(
      "lm_head" + model_postfix
    );
  }

  auto rms_norm = RMSNorm(config.embed_dim);
  rms_norm->weight = st_loader.findWeight(
    model_prefix + "norm" + model_postfix
  );

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
  c10::InferenceMode guard;

  if (use_cache) {
    shard_decoder->setup_caches(
      1,
      config.max_seq_len
    );
  }

  return shard_decoder->forward(
    tokens_,
    mask_,
    input_pos_,
    hidden_state_
  );
}