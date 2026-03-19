from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).pow(2)


class MambaMLP(nn.Module):
    def __init__(self, config: dict, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = int(config["embed_dim"])
        pattern = str(config.get("hybrid_override_pattern", ""))
        if isinstance(config.get("intermediate_dim"), list):
            mlp_idx = pattern[: (layer_idx or 0) + 1].count("-") - 1
            dims = config["intermediate_dim"]
            if len(dims) == 1:
                intermediate_size = int(dims[0])
            else:
                intermediate_size = int(dims[max(0, mlp_idx)])
        else:
            intermediate_size = int(config.get("intermediate_dim", self.hidden_size * 4))

        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=bool(config.get("mlp_bias", False)))
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=bool(config.get("mlp_bias", False)))
        activation_name = str(config.get("mlp_hidden_act", config.get("hidden_act", "relu"))).lower()
        if activation_name in ACT2FN:
            self.act_fn = ACT2FN[activation_name]
        elif activation_name == "relu2":
            self.act_fn = _relu2
        else:
            self.act_fn = torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))
