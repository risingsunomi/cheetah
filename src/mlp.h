// Multilayer Perceptron (MLP)
#ifndef MLP_H
#define MLP_H

#include <torch/torch.h>

class MLPImpl : public torch::nn::Module {
    public:
        MLPImpl(int embed_dim, int hidden_dim);
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Linear gate_proj{nullptr};
        torch::nn::Linear up_proj{nullptr};
        torch::nn::Linear down_proj{nullptr};
};
TORCH_MODULE(MLP);

#endif // MLP_H