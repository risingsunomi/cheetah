// General Multi-Head Attention Model
// Generalized model for use with LLMs and other multi-head attention models
#ifndef GENERAL_MHA_MODEL_H
#define GENERAL_MHA_MODEL_H

#include <torch/torch.h>
#include "rope.h"
#include "attention.h"
#include "cache.h"
#include "rms.h"
#include "mlp.h"
#include "transformer.h"

class GeneralMHAModelImpl : public torch::nn::Module {
    public:
        GeneralMHAModelImpl(
            int64_t layer_start,
            int64_t layer_end,
            int64_t layer_total,
            int64_t vocab_size,
            int64_t embed_dim,
            int64_t hidden_dim,
            int64_t num_heads,
            int64_t num_kv_heads,
            int64_t head_dim,
            int64_t max_seq_len
        );
        torch::Tensor forward(
            const torch::Tensor& tokens,
            const c10::optional<torch::Tensor>& mask,
            const c10::optional<torch::Tensor>& encoder_input,
            const c10::optional<torch::Tensor>& encoder_mask,
            const c10::optional<torch::Tensor>& input_pos
        );

    private:
        torch::nn::Embedding tok_embeddings{nullptr};
        torch::nn::Linear output_proj{nullptr};
        TransformerDecoder decoder = nullptr;
        RMSNorm final_norm = nullptr;
        int64_t layer_start;
        int64_t layer_end;
        int64_t layer_total;
};
TORCH_MODULE(GeneralMHAModel);

#endif // GENERAL_MHA_MODEL_H