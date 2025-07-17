#ifndef RMS_NORM_H
#define RMS_NORM_H

#include <torch/torch.h>

class RMSNormImpl : public torch::nn::Module {
    public:
        RMSNormImpl(int hidden_size, float eps = 1e-5);
        torch::Tensor forward(const torch::Tensor& input);
        torch::Tensor weight;

    private:
        float eps;
};
TORCH_MODULE(RMSNorm);

#endif // RMS_NORM_H