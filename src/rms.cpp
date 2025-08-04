#include "rms.h"

RMSNormImpl::RMSNormImpl(int hidden_size_, float eps_)
    : eps(eps_) {
  weight = register_parameter("weight", torch::ones({hidden_size_}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& input_) {
  auto norm_x = input_.norm(2, -1, true);
  return (input_ / (norm_x / std::sqrt(input_.size(-1)) + eps)) * weight;
}