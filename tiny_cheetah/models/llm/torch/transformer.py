from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .attention import MultiHeadAttention
from .mlp import MLP
from .moe import MOEMLP


class _FallbackRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


def _rms_norm(dim: int, eps: float = 1e-6) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim, eps=eps)
    return _FallbackRMSNorm(dim, eps=eps)


def _scale_output(scale: Any, value: torch.Tensor) -> torch.Tensor:
    if scale is None:
        return value
    if callable(scale):
        return scale(value)
    if isinstance(scale, (int, float)):
        return value * float(scale)
    return value


class TransformerBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int | None = None):
        super().__init__()
        norm_eps = config.get("norm_eps", 1e-6)
        self.input_layernorm = _rms_norm(config["embed_dim"], eps=norm_eps)
        if bool(config.get("moe")):
            self.mlp = MOEMLP(config)
        else:
            self.mlp = MLP(
                config.get("embed_dim"),
                config.get("intermediate_dim"),
                config.get("hidden_act"),
                config.get("mlp_bias", True),
            )
        self.self_attn = MultiHeadAttention(
            config=config,
            is_causal=any("CausalLM" in arch for arch in config.get("architectures", [])),
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = _rms_norm(config.get("embed_dim"), eps=norm_eps)

        self.attn_scale = config.get("attn_scale")
        self.mlp_scale = config.get("mlp_scale")

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, attention_mask, position_ids)
        x = x + _scale_output(self.attn_scale, attn_out)

        mlp_out = self.mlp(self.post_attention_layernorm(x))
        return x + _scale_output(self.mlp_scale, mlp_out)
