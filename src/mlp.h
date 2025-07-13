// Multilayer Perceptron (MLP)
#ifndef MLP_H
#define MLP_H

#include <torch/torch.h>

const std::string HIDDEN_ACTIVATION = "silu";

class MLPImpl : public torch::nn::Module {
    public:
        MLPImpl(
            int embed_dim_,
            int hidden_dim_,
            const std::string& hidden_act_ = HIDDEN_ACTIVATION
        );
        torch::Tensor forward(const torch::Tensor& x_);

    private:
        int embed_dim;
        int hidden_dim;
        const std::string& hidden_act;
        torch::nn::Linear gate_proj{nullptr};
        torch::nn::Linear up_proj{nullptr};
        torch::nn::Linear down_proj{nullptr};
};
TORCH_MODULE(MLP);

#endif // MLP_H