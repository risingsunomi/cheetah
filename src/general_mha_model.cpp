#include "general_mha_model.h"

GeneralMHAModel::GeneralMHAModel(
    const Shard& shard_,
    const ModelConfig& config_,
    const std::string safetensors_path_,
    bool& use_cache_
) : shard(shard_),
    config(config_),
    use_cache(use_cache_),
    safetensors_path(safetensors_path_){

    // load safetensors
    std::cout << "generlmha constructor called" << std::endl;
    std::cout << "Loading safetensors from " + safetensors_path_ << std::endl;
    SafeTensorsLoader safetensors_loader(safetensors_path_);
    auto model_weights = safetensors_loader.getTensors();

    // will make post and prefixes customizable but for now hard coded
    std::string model_prefix = "model."; 
    std::string model_layer_prefix = "model.layers.";
    std::string model_self_attention_label = "self_attn";
    std::string model_postfix = ".weight";

    auto norm_weight = model_weights.at(
        model_prefix + "norm" + model_postfix);

    auto rms_norm = RMSNorm(config.embed_dim);
    rms_norm->weight = norm_weight;

    // Create decoder layers from layer_start to layer_end
    for (int i = shard.start_layer; i < shard.end_layer; ++i) {
        // Instantiate TransformerSelfAttentionLayer
        auto q_proj = torch::nn::Linear(config.embed_dim, config.embed_dim);
        
        auto transformer_layer = register_module(
            "transformer_self_attention_layer_" + std::to_string(i),
            TransformerSelfAttentionLayer(
                MultiHeadAttention(
                    i,
                    config.embed_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    q_proj, // q_proj
                    torch::nn::Linear(config.embed_dim, config.embed_dim), // k_proj
                    torch::nn::Linear(config.embed_dim, config.embed_dim), // v_proj
                    torch::nn::Linear(config.embed_dim, config.embed_dim), // out_proj
                    RotaryEmbedding(
                        config.head_dim, 
                        config.max_seq_len,
                        config.rope_base
                    ),
                    c10::nullopt,
                    true,
                    config.attn_dropout,
                    use_cache
                ),
                MLP(
                    config.embed_dim,
                    config.intermediate_size,
                    config.hidden_act
                ),
                torch::nn::AnyModule(rms_norm),
                torch::nn::AnyModule(rms_norm)
            )
        );
        
        self_attn_layers.push_back(
            register_module("layer_" + std::to_string(i), transformer_layer));
    }

    shard_decoder = ShardTransformerDecoder(
        shard,
        torch::nn::Embedding(config.vocab_size, config.embed_dim),
        self_attn_layers,
        config.max_seq_len,
        rms_norm,
        torch::nn::Linear(config.embed_dim, config.vocab_size)
    );
}

torch::Tensor GeneralMHAModel::forward(
    const torch::Tensor& tokens_,
    const c10::optional<torch::Tensor> mask_,
    const c10::optional<torch::Tensor> input_pos_,
    const c10::optional<torch::Tensor> hidden_state_
) {
    if(use_cache) {
        // fix this in model_config.h
        // if(config.torch_dtype == "bfloat32") {
        //     shard_decoder->setup_caches(
        //         1,
        //         torch::kBFloat16,
        //         config.max_seq_len
        //     );
        // } else {
        shard_decoder->setup_caches(
            1,
            torch::kFloat32,
            config.max_seq_len
        );
        // }
        
    }
    return shard_decoder->forward(tokens_, mask_, input_pos_, hidden_state_);
}
