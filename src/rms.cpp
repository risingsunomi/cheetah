#include "rms.h"

RMSNormImpl::RMSNormImpl(int hidden_size, float eps)
    : eps(eps) {
  weight = register_parameter("weight", torch::ones({hidden_size}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& input) {
  auto norm_x = input.norm(2, -1, true);
  return (input / (norm_x / std::sqrt(input.size(-1)) + eps)) * weight;
}