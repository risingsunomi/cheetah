#include <torch/torch.h>
#include "rope.h"
#include "attention.h"
#include "cache.h"
#include "rms.h"
#include "mlp.h"

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