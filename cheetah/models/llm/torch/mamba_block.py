from __future__ import annotations

import torch
from torch import nn

from .mamba_attention import MambaAttention
from .mamba_mixer import MambaMixer, MambaStateCache
from .mamba_mlp import MambaMLP


class _FallbackRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x).to(input_dtype)


def _rms_norm(dim: int, eps: float = 1e-6) -> nn.Module:
    return _FallbackRMSNorm(dim, eps=eps)


class MambaHybridBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = bool(config.get("residual_in_fp32", False))
        self.norm = _rms_norm(int(config["embed_dim"]), eps=float(config.get("layer_norm_epsilon", config.get("norm_eps", 1e-6))))
        layer_types = list(config.get("layers_block_type", []))
        self.block_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "mlp"

        if self.block_type == "mamba":
            self.mixer = MambaMixer(config, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = MambaAttention(config, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = MambaMLP(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unsupported Mamba hybrid block type: {self.block_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        mamba_cache: MambaStateCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states,
                cache_params=mamba_cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        else:
            hidden_states = self.mixer(hidden_states)

        return residual + hidden_states
