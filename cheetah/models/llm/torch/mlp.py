from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """
    SwiGLU-style MLP block.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.relu(x)
        if self.activation == "silu":
            return torch.nn.functional.silu(x)
        if self.activation == "gelu":
            return torch.nn.functional.gelu(x)
        if self.activation == "selu":
            return torch.nn.functional.selu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))
