// general_mha.h
#pragma once

#include <torch/torch.h>
#include "rope.h"

class RMSNormImpl : public torch::nn::Module {
    public:
        RMSNormImpl(int64_t hidden_size, float eps = 1e-5);
        torch::Tensor forward(const torch::Tensor& input);

    private:
        torch::Tensor weight;
        float eps;
};
TORCH_MODULE(RMSNorm);

class MLPImpl : public torch::nn::Module {
    public:
        MLPImpl(int64_t input_dim, int64_t hidden_dim);
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Linear gate_proj{nullptr};
        torch::nn::Linear up_proj{nullptr};
        torch::nn::Linear down_proj{nullptr};
};
TORCH_MODULE(MLP);

class MultiHeadAttentionImpl : public torch::nn::Module {
    public:
        MultiHeadAttentionImpl(
            int64_t embed_dim,
            int64_t num_heads,
            int64_t head_dim,
            int64_t max_seq_len
        );

        torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

    private:
        torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, out_proj{nullptr};
        int64_t num_heads;
        int64_t head_dim;
        RotaryEmbedding rope;
};
TORCH_MODULE(MultiHeadAttention);

class TransformerBlockImpl : public torch::nn::Module {
    public:
        TransformerBlockImpl(
            int64_t embed_dim,
            int64_t hidden_dim,
            int64_t num_heads,
            int64_t head_dim,
            int64_t max_seq_len
        );
        torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

    private:
        MultiHeadAttention mha;
        MLP mlp;
        RMSNorm norm1, norm2;
};
TORCH_MODULE(TransformerBlock);

class GeneralMHAImpl : public torch::nn::Module {
    public:
        GeneralMHAImpl(
            int64_t vocab_size,
            int64_t embed_dim,
            int64_t hidden_dim,
            int64_t num_heads,
            int64_t head_dim,
            int64_t max_seq_len
        );
        torch::Tensor forward(torch::Tensor tokens, torch::Tensor mask = {});

    private:
        torch::nn::Embedding tok_embeddings{nullptr};
        torch::nn::Linear output_proj{nullptr};
        TransformerBlock block;
        RMSNorm final_norm;
};
TORCH_MODULE(GeneralMHA);