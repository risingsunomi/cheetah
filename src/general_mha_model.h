// General Multi-Head Attention Model
// Generalized model for use with LLMs and other multi-head attention models
#ifndef GENERAL_MHA_MODEL_H
#define GENERAL_MHA_MODEL_H

#include <torch/torch.h>
#include "rope.h"
#include "attention.h"
#include "utils/cache.h"
#include "rms.h"
#include "mlp.h"
#include "transformers.h"
#include "utils/shard.h"
#include "utils/model_config.h"
#include "utils/safetensors_loader.h"
#include "utils/helpers.h"

class GeneralMHAModel : public torch::nn::Module {
public:
  GeneralMHAModel(
      Shard shard_,
      ModelConfig config_,
      const std::string model_path_,
      bool& use_cache_
  );
  torch::Tensor forward(
      const torch::Tensor& tokens_,
      const c10::optional<torch::Tensor> mask_,
      const c10::optional<torch::Tensor> input_pos_,
      const c10::optional<torch::Tensor> hidden_state_
  );

  std::vector<TransformerSelfAttentionLayer> self_attn_layers;
  ShardTransformerDecoder shard_decoder = nullptr;
  const std::string model_path;
  Shard shard;
  ModelConfig config;
  bool use_cache;

private:
  
  std::string model_prefix = "model.";
  std::string model_layer_prefix = "model.layers.";
  std::string model_postfix = ".weight";
};

#endif // GENERAL_MHA_MODEL_H