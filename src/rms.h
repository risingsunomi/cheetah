#ifndef RMS_H
#define RMS_H

#include <torch/torch.h>

class RMSNormImpl : public torch::nn::Module {
    public:
        RMSNormImpl(int hidden_size_, float eps_ = 1e-5);
        torch::Tensor forward(const torch::Tensor& input_);
        torch::Tensor weight;

    private:
        float eps;
};
TORCH_MODULE(RMSNorm);

#endif // RMS_H