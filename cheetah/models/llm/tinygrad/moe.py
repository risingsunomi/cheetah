from __future__ import annotations

import numpy as np
import tinygrad as tg


def _decode_fp4_values(indices: np.ndarray) -> np.ndarray:
    values = indices.astype(np.int32)
    magnitude = values & 0x07
    sign = np.where((values & 0x08) == 0, 1.0, -1.0).astype(np.float32)

    decoded = np.where(
        magnitude == 0,
        0.0,
        np.where(magnitude < 5, magnitude.astype(np.float32) * 0.5, magnitude.astype(np.float32) - 2.0),
    ).astype(np.float32)
    decoded = np.where(magnitude == 7, 6.0, decoded).astype(np.float32)
    return decoded * sign


class MOERouter:
    def __init__(self, embed_dim: int, num_experts: int, top_k: int) -> None:
        self.hidden_dim = int(embed_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.weight = tg.Tensor.zeros((self.num_experts, self.hidden_dim), dtype=tg.dtypes.float32)
        self.bias = tg.Tensor.zeros((self.num_experts,), dtype=tg.dtypes.float32)

    def __call__(self, hidden_states: tg.Tensor) -> tuple[tg.Tensor, tg.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = flat.numpy().astype(np.float32) @ self.weight.numpy().astype(np.float32).T
        logits += self.bias.numpy().astype(np.float32)

        top_k = min(self.top_k, self.num_experts)
        if top_k <= 0:
            empty = np.empty((logits.shape[0], 0), dtype=np.int32)
            zero = np.zeros_like(logits, dtype=np.float32)
            return (
                tg.Tensor(zero, device=hidden_states.device, dtype=tg.dtypes.float32),
                tg.Tensor(empty, device=hidden_states.device, dtype=tg.dtypes.int32),
            )

        indices = np.argpartition(-logits, top_k - 1, axis=-1)[:, :top_k]
        values = np.take_along_axis(logits, indices, axis=-1)
        order = np.argsort(-values, axis=-1)
        indices = np.take_along_axis(indices, order, axis=-1).astype(np.int32)
        values = np.take_along_axis(values, order, axis=-1)

        values = np.exp(values - values.max(axis=-1, keepdims=True))
        values /= values.sum(axis=-1, keepdims=True)

        scores = np.zeros_like(logits, dtype=np.float32)
        np.put_along_axis(scores, indices, values.astype(np.float32), axis=-1)
        return (
            tg.Tensor(scores, device=hidden_states.device, dtype=tg.dtypes.float32),
            tg.Tensor(indices, device=hidden_states.device, dtype=tg.dtypes.int32),
        )


class MOEExperts:
    def __init__(self, num_experts: int, embed_dim: int, hidden_dim: int) -> None:
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.groups_in = max(1, self.embed_dim // 32)
        self.groups_hidden = max(1, self.hidden_dim // 32)

        self.gate_up_proj_blocks = tg.Tensor.zeros(
            (self.num_experts, self.hidden_dim * 2, self.groups_in, 16),
            dtype=tg.dtypes.uint8,
        )
        self.gate_up_proj_scales = tg.Tensor.zeros(
            (self.num_experts, self.hidden_dim * 2, self.groups_in),
            dtype=tg.dtypes.uint8,
        )
        self.gate_up_proj_bias = tg.Tensor.zeros(
            (self.num_experts, self.hidden_dim * 2),
            dtype=tg.dtypes.float32,
        )
        self.down_proj_blocks = tg.Tensor.zeros(
            (self.num_experts, self.embed_dim, self.groups_hidden, 16),
            dtype=tg.dtypes.uint8,
        )
        self.down_proj_scales = tg.Tensor.zeros(
            (self.num_experts, self.embed_dim, self.groups_hidden),
            dtype=tg.dtypes.uint8,
        )
        self.down_proj_bias = tg.Tensor.zeros(
            (self.num_experts, self.embed_dim),
            dtype=tg.dtypes.float32,
        )

    def _decode_mxfp4(self, blocks: tg.Tensor, scales: tg.Tensor) -> tg.Tensor:
        blocks_np = blocks.numpy().astype(np.uint8)
        scales_np = scales.numpy().astype(np.int32) - 127

        idx_lo = blocks_np & 0x0F
        idx_hi = blocks_np >> 4
        decoded = np.empty((*blocks_np.shape[:-1], blocks_np.shape[-1] * 2), dtype=np.float32)
        decoded[..., 0::2] = _decode_fp4_values(idx_lo)
        decoded[..., 1::2] = _decode_fp4_values(idx_hi)
        decoded = np.ldexp(decoded, np.expand_dims(scales_np, axis=-1))
        decoded = decoded.reshape(blocks_np.shape[0], -1)
        return tg.Tensor(decoded.astype(np.float32), device=blocks.device, dtype=tg.dtypes.float32)

    @staticmethod
    def _gpt_oss_swiglu(gate_up: np.ndarray, limit: float) -> np.ndarray:
        gate = np.clip(gate_up[..., ::2], a_min=None, a_max=float(limit))
        linear = np.clip(gate_up[..., 1::2], a_min=-float(limit), a_max=float(limit))
        return gate * (1.0 / (1.0 + np.exp(-1.702 * gate))) * (linear + 1.0)

    def run_expert(self, hidden: tg.Tensor, expert_idx: int, swiglu_limit: float) -> tg.Tensor:
        hidden_np = hidden.numpy().astype(np.float32)

        gate_up_weight = self._decode_mxfp4(
            self.gate_up_proj_blocks[expert_idx],
            self.gate_up_proj_scales[expert_idx],
        ).numpy()
        gate_up_bias = self.gate_up_proj_bias[expert_idx].numpy().astype(np.float32)
        gate_up = hidden_np @ gate_up_weight.T + gate_up_bias
        activated = self._gpt_oss_swiglu(gate_up, limit=swiglu_limit)

        down_weight = self._decode_mxfp4(
            self.down_proj_blocks[expert_idx],
            self.down_proj_scales[expert_idx],
        ).numpy()
        down_bias = self.down_proj_bias[expert_idx].numpy().astype(np.float32)
        out = activated @ down_weight.T + down_bias
        return tg.Tensor(out.astype(np.float32), device=hidden.device, dtype=tg.dtypes.float32)

    def __call__(
        self,
        hidden_states: tg.Tensor,
        router_indices: tg.Tensor,
        routing_weights: tg.Tensor,
        swiglu_limit: float,
    ) -> tg.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_dim)
        flat_states_np = flat_states.numpy().astype(np.float32)
        mixed = np.zeros((flat_states.shape[0], self.embed_dim), dtype=np.float32)

        router_indices_np = router_indices.numpy().astype(np.int64)
        if router_indices_np.size == 0:
            return tg.Tensor(mixed.reshape(*original_shape[:-1], self.embed_dim), device=hidden_states.device)

        routing_weights_np = routing_weights.numpy().astype(np.float32)
        for expert_idx in np.unique(router_indices_np):
            token_idx = np.nonzero(np.any(router_indices_np == int(expert_idx), axis=1))[0]
            if token_idx.size == 0:
                continue

            current_state = tg.Tensor(flat_states_np[token_idx], device=hidden_states.device, dtype=tg.dtypes.float32)
            expert_out = self.run_expert(
                current_state,
                expert_idx=int(expert_idx),
                swiglu_limit=swiglu_limit,
            ).numpy().astype(np.float32)
            weights = routing_weights_np[token_idx, int(expert_idx)][:, None]
            mixed[token_idx] += expert_out * weights

        return tg.Tensor(mixed.reshape(*original_shape[:-1], self.embed_dim), device=hidden_states.device)


class MOEMLP:
    def __init__(self, config: dict):
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

    def __call__(self, x: tg.Tensor) -> tg.Tensor:
        if self.num_experts <= 0 or self.experts_per_token <= 0:
            return tg.Tensor.zeros(x.shape, device=x.device, dtype=x.dtype)

        routing_weights, router_indices = self.router(x)
        return self.experts(
            x,
            router_indices=router_indices,
            routing_weights=routing_weights,
            swiglu_limit=self.swiglu_limit,
        ).cast(x.dtype)
