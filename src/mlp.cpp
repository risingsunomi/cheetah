#include "mlp.h"

MLPImpl::MLPImpl(
  int embed_dim_,
  int hidden_dim_,
  const std::string& hidden_act_
) : embed_dim(embed_dim_),
    hidden_dim(hidden_dim_),
    hidden_act(std::move(hidden_act_)) {
  gate_proj = register_module(
    "gate_proj",
    torch::nn::Linear(
      torch::nn::LinearOptions(
        embed_dim, hidden_dim
      ).bias(false)
    )
  );

  up_proj = register_module(
    "up_proj",
    torch::nn::Linear(
      torch::nn::LinearOptions(
        embed_dim, hidden_dim
      ).bias(false)
    )
  );

  down_proj = register_module(
    "down_proj",
    torch::nn::Linear(
      torch::nn::LinearOptions(
        hidden_dim, embed_dim
      ).bias(false)
    )
  );
}

torch::Tensor MLPImpl::forward(const torch::Tensor& x_) {
  torch::Tensor gate;

  if (hidden_act == "silu"){
    gate = torch::silu(gate_proj->forward(x_));
  } else if (hidden_act == "relu") {
    gate = torch::relu(gate_proj->forward(x_));
  } else if (hidden_act == "gelu") {
    gate = torch::gelu(gate_proj->forward(x_));
  } else if (hidden_act == "tanh") {
    gate = torch::tanh(gate_proj->forward(x_));
  } else if (hidden_act == "sigmoid") {
    gate = torch::sigmoid(gate_proj->forward(x_));
  } else if (hidden_act == "leaky_relu") {
    gate = torch::leaky_relu(gate_proj->forward(x_), 0.2);
  } else {
    gate = torch::silu(gate_proj->forward(x_));
  }

  auto up = up_proj->forward(x_);
  return down_proj->forward(gate * up);
}