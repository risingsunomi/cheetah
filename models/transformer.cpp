#include "transformer.h"

TransformerDecoderImpl::TransformerDecoderImpl(
    torch::nn::Embedding tok_embeddings,
    std::vector<torch::nn::ModuleHolder<MultiHeadAttentionImpl>> layers,
    int64_t max_seq_len,
    int64_t num_heads,
    int64_t head_dim,
    torch::nn::ModuleHolder<torch::nn::Module> norm,
    torch::nn::ModuleHolder<torch::nn::Module> output,
    std::vector<int> output_hidden_states
) :
    tok_embeddings(tok_embeddings),
    layers(std::move(layers)),
    max_seq_len(max_seq_len),
    num_heads(num_heads),
    head_dim(head_dim),
    norm(norm),
    output(output),
    output_hidden_states(std::move(output_hidden_states))
{
    register_module("tok_embeddings", tok_embeddings);
    register_module("norm", norm);
    register_module("output", output);

    for (size_t i = 0; i < layers.size(); ++i) {
        register_module("layer_" + std::to_string(i), layers[i]);
    }
}

void TransformerDecoderImpl::setup_caches(
    int64_t batch_size, torch::Dtype dtype,
    c10::optional<int64_t> encoder_max_seq_len,
    c10::optional<int64_t> decoder_max_seq_len) {
        encoder_max_cache_seq_len = encoder_max_seq_len.value_or(max_seq_len);
        decoder_max_cache_seq_len = decoder_max_seq_len.value_or(max_seq_len);

    for (auto& layer : layers) {
        layer->setup_cache(
            batch_size,
            dtype,
            encoder_max_cache_seq_len,
            decoder_max_cache_seq_len
        );
    }
}

bool TransformerDecoderImpl::caches_are_setup() const {
    return layers[0]->kv_cache &&
           layers[0]->kv_cache->k_cache.defined() &&
           layers[0]->kv_cache->v_cache.defined();
}

bool TransformerDecoderImpl::caches_are_enabled() const {
    return layers[0]->is_cache_enabled;
}

void TransformerDecoderImpl::reset_caches() {
    if (!caches_are_enabled()) {
        throw std::runtime_error("KV caches not setup");
    }

    for (auto& layer : layers) {
        layer->reset_cache();
    }
}

void TransformerDecoderImpl::set_num_output_chunks(int64_t num_chunks) {
    num_output_chunks = num_chunks;
}

torch::Tensor TransformerDecoderImpl::forward(
    const torch::Tensor& tokens,
    const c10::optional<torch::Tensor>& mask,
    const c10::optional<torch::Tensor>& encoder_input,
    const c10::optional<torch::Tensor>& encoder_mask,
    const c10::optional<torch::Tensor>& input_pos
) {
    int64_t seq_len = tokens.size(1);
    if (seq_len > max_seq_len) {
        throw std::runtime_error("Input seq_len > max_seq_len");
    }

    if (caches_are_enabled()) {
        if (!mask.has_value()) throw std::runtime_error("Mask required for inference with caches");
        if (encoder_input.has_value() && !encoder_mask.has_value()) throw std::runtime_error("Encoder mask required");
        if (!input_pos.has_value()) throw std::runtime_error("input_pos required when using caches");
    }

    torch::Tensor h = tok_embeddings->forward(tokens);
    std::vector<torch::Tensor> hidden;

    for (size_t i = 0; i < layers.size(); ++i) {
        if (std::find(output_hidden_states.begin(), output_hidden_states.end(), i) != output_hidden_states.end()) {
            hidden.push_back(h);
        }
        h = layers[i]->forward(h, mask, encoder_input, encoder_mask, input_pos);
    }

    h = norm->forward(h);

    torch::Tensor out;
    if (num_output_chunks > 0) {
        auto chunks = h.chunk(num_output_chunks, 1);
        std::vector<torch::Tensor> outputs;
        for (auto& c : chunks) {
            outputs.push_back(output->forward(c));
        }
        out = torch::cat(outputs, 1);
    } else {
        out = output->forward(h).to(torch::kFloat32);
    }

    if (!hidden.empty()) {
        hidden.push_back(out);
        return torch::stack(hidden);
    } else {
        return out;
    }
}
