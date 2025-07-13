#include "mlp.h"

MLPImpl::MLPImpl(int embed_dim, int hidden_dim) {
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
        embed_dim,hidden_dim
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

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  auto gate = torch::silu(gate_proj->forward(x));
  auto up = up_proj->forward(x);
  return down_proj->forward(gate * up);
}