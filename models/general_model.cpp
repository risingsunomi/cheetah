#include "general_model.h"
#include "transformer.h"
#include "attention.h"

GeneralMHAModelImpl::GeneralMHAModelImpl(
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
) : layer_start(layer_start), layer_end(layer_end), layer_total(layer_total) {
    // Token embedding layer
    tok_embeddings = register_module(
        "tok_embeddings",
        torch::nn::Embedding(vocab_size, embed_dim)
    );

    // Output projection (usually vocab size)
    output_proj = register_module(
        "output_proj",
        torch::nn::Linear(torch::nn::LinearOptions(embed_dim, vocab_size).bias(false))
    );

    // Create decoder layers from layer_start to layer_end
    std::vector<MultiHeadAttention> decoder_layers;
    for (int64_t i = layer_start; i < layer_end; ++i) {
        auto layer = MultiHeadAttention(
            embed_dim,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len
        );
        
        decoder_layers.push_back(
            register_module("decoder_layer_" + std::to_string(i), layer));
    }

    // Normalization before output
    auto norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}));
    register_module("norm", norm);

    // Instantiate TransformerDecoder
    decoder = register_module(
        "decoder",
        TransformerDecoder(
            tok_embeddings,
            decoder_layers,
            max_seq_len,
            num_heads,
            head_dim,
            norm,
            output_proj
        )
    );
}

torch::Tensor GeneralMHAModelImpl::forward(
    const torch::Tensor& tokens,
    const c10::optional<torch::Tensor>& mask,
    const c10::optional<torch::Tensor>& encoder_input,
    const c10::optional<torch::Tensor>& encoder_mask,
    const c10::optional<torch::Tensor>& input_pos
) {
    return decoder->forward(tokens, mask, encoder_input, encoder_mask, input_pos);
}
