#ifndef CHEETAH_MODELS_TRANSFORMER_H
#define CHEETAH_MODELS_TRANSFORMER_H

#include <torch/torch.h>
#include <vector>
#include <memory>
#include "attention.h"
#include "cache.h"

class TransformerDecoderImpl : public torch::nn::Module {
public:
    TransformerDecoderImpl(
        torch::nn::Embedding tok_embeddings,
        std::vector<torch::nn::ModuleHolder<MultiHeadAttentionImpl>> layers,
        int64_t max_seq_len,
        int64_t num_heads,
        int64_t head_dim,
        torch::nn::ModuleHolder<torch::nn::Module> norm,
        torch::nn::ModuleHolder<torch::nn::Module> output,
        std::vector<int> output_hidden_states = {}
    );

    torch::Tensor forward(
        const torch::Tensor& tokens,
        const c10::optional<torch::Tensor>& mask = c10::nullopt,
        const c10::optional<torch::Tensor>& encoder_input = c10::nullopt,
        const c10::optional<torch::Tensor>& encoder_mask = c10::nullopt,
        const c10::optional<torch::Tensor>& input_pos = c10::nullopt
    );

    void setup_caches(
        int64_t batch_size,
        torch::Dtype dtype,
        c10::optional<int64_t> encoder_max_seq_len = c10::nullopt,
        c10::optional<int64_t> decoder_max_seq_len = c10::nullopt
    );

    void reset_caches();
    bool caches_are_setup() const;
    bool caches_are_enabled() const;

    void set_num_output_chunks(int64_t num_chunks);

private:
    torch::nn::Embedding tok_embeddings;
    std::vector<torch::nn::ModuleHolder<MultiHeadAttentionImpl>> layers;
    torch::nn::ModuleHolder<torch::nn::Module> norm;
    torch::nn::ModuleHolder<torch::nn::Module> output;

    std::vector<int> output_hidden_states;
    int64_t max_seq_len;
    int64_t num_heads;
    int64_t head_dim;
    int64_t num_output_chunks = 0;

    int64_t encoder_max_cache_seq_len = -1;
    int64_t decoder_max_cache_seq_len = -1;
};
TORCH_MODULE(TransformerDecoder);

#endif // CHEETAH_MODELS_TRANSFORMER_H