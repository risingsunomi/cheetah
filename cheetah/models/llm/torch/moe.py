from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _decode_fp4_values(indices: torch.Tensor) -> torch.Tensor:
    values = indices.to(torch.int32)
    magnitude = values & 0x07
    sign = torch.where(
        (values & 0x08) == 0,
        torch.ones_like(magnitude, dtype=torch.float32),
        -torch.ones_like(magnitude, dtype=torch.float32),
    )

    decoded = torch.where(
        magnitude == 0,
        torch.zeros_like(sign),
        torch.where(
            magnitude < 5,
            magnitude.to(torch.float32) * 0.5,
            magnitude.to(torch.float32) - 2.0,
        ),
    )
    decoded = torch.where(magnitude == 7, torch.full_like(decoded, 6.0), decoded)
    return decoded * sign


class MOERouter(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.hidden_dim = int(embed_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(flat.to(self.weight.dtype), self.weight, self.bias)

        top_k = min(self.top_k, self.num_experts)
        if top_k <= 0:
            empty = torch.empty((flat.shape[0], 0), device=flat.device, dtype=torch.long)
            return torch.zeros_like(router_logits), empty

        router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)
        # GPT-OSS normalizes only over the selected experts.
        router_top_value = torch.softmax(
            router_top_value,
            dim=-1,
            dtype=router_top_value.dtype,
        )
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class MOEExperts(nn.Module):
    """
    GPT-OSS-style packed MXFP4 experts.

    gate_up_proj_* and down_proj_* tensors are stored in packed block format:
    - *_blocks: uint8 packed nibbles (two FP4 values per byte)
    - *_scales: uint8 exponent offsets (E8M0-like, minus 127 bias)
    """

    def __init__(self, num_experts: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.groups_in = max(1, self.embed_dim // 32)
        self.groups_hidden = max(1, self.hidden_dim // 32)
        self.alpha = 1.702

        self.gate_up_proj_blocks = nn.Parameter(
            torch.empty(
                (self.num_experts, self.hidden_dim * 2, self.groups_in, 16),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_scales = nn.Parameter(
            torch.empty(
                (self.num_experts, self.hidden_dim * 2, self.groups_in),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty((self.num_experts, self.hidden_dim * 2), dtype=torch.float32),
            requires_grad=False,
        )

        self.down_proj_blocks = nn.Parameter(
            torch.empty(
                (self.num_experts, self.embed_dim, self.groups_hidden, 16),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj_scales = nn.Parameter(
            torch.empty(
                (self.num_experts, self.embed_dim, self.groups_hidden),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty((self.num_experts, self.embed_dim), dtype=torch.float32),
            requires_grad=False,
        )

    def _decode_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        idx_lo = blocks & 0x0F
        idx_hi = blocks >> 4

        decoded = torch.empty(
            (*blocks.shape[:-1], blocks.shape[-1] * 2),
            dtype=dtype,
            device=blocks.device,
        )
        decoded[..., 0::2] = _decode_fp4_values(idx_lo).to(dtype=dtype)
        decoded[..., 1::2] = _decode_fp4_values(idx_hi).to(dtype=dtype)
        exponents = scales.to(torch.int32) - 127
        decoded = torch.ldexp(decoded, exponents.unsqueeze(-1))
        return decoded.reshape(blocks.shape[0], -1)

    @staticmethod
    def _gpt_oss_swiglu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
        # GPT-OSS stores gate/up values interleaved in the last dimension.
        gate, linear = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=float(limit))
        linear = linear.clamp(min=-float(limit), max=float(limit))
        return gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)

    def run_expert(self, hidden: torch.Tensor, expert_idx: int, swiglu_limit: float) -> torch.Tensor:
        compute_dtype = hidden.dtype if hidden.dtype.is_floating_point else torch.float32
        hidden_f = hidden.to(compute_dtype)

        gate_up_weight = self._decode_mxfp4(
            self.gate_up_proj_blocks[expert_idx],
            self.gate_up_proj_scales[expert_idx],
            dtype=compute_dtype,
        )
        gate_up_bias = self.gate_up_proj_bias[expert_idx].to(compute_dtype)
        gate_up = F.linear(hidden_f, gate_up_weight, gate_up_bias)
        activated = self._gpt_oss_swiglu(gate_up, limit=swiglu_limit)

        down_weight = self._decode_mxfp4(
            self.down_proj_blocks[expert_idx],
            self.down_proj_scales[expert_idx],
            dtype=compute_dtype,
        )
        down_bias = self.down_proj_bias[expert_idx].to(compute_dtype)
        return F.linear(activated, down_weight, down_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        swiglu_limit: float,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_dim)
        mixed = torch.zeros(
            (flat_states.shape[0], self.embed_dim),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        if router_indices.numel() == 0:
            return mixed.reshape(*original_shape[:-1], self.embed_dim).to(hidden_states.dtype)

        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit[:, 0]:
            _, token_idx = torch.where(expert_mask[int(expert_idx)])
            current_state = flat_states[token_idx]
            expert_out = self.run_expert(
                current_state,
                expert_idx=int(expert_idx),
                swiglu_limit=swiglu_limit,
            )
            weighted_output = expert_out * routing_weights[token_idx, int(expert_idx), None].to(torch.float32)
            mixed.index_add_(0, token_idx, weighted_output)

        return mixed.reshape(*original_shape[:-1], self.embed_dim).to(hidden_states.dtype)


class MOEMLP(nn.Module):
    """
    Top-k routed MoE MLP using packed MXFP4 expert tensors.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = int(config["embed_dim"])
        self.hidden_dim = int(config["intermediate_dim"])
        self.num_experts = int(config.get("num_local_experts", 0))
        self.experts_per_token = int(config.get("num_experts_per_tok", 0))
        self.swiglu_limit = float(config.get("swiglu_limit", 7.0))
        self.router = MOERouter(
            embed_dim=self.embed_dim,
            num_experts=self.num_experts,
            top_k=self.experts_per_token,
        )
        self.experts = MOEExperts(
            num_experts=self.num_experts,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_experts <= 0 or self.experts_per_token <= 0:
            return torch.zeros_like(x)

        routing_weights, router_indices = self.router(x)
        return self.experts(
            x,
            router_indices=router_indices,
            routing_weights=routing_weights,
            swiglu_limit=self.swiglu_limit,
        )
