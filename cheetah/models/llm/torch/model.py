from __future__ import annotations

import torch
from torch import nn

from cheetah.models.llm.backend import get_backend_device
from ...shard import Shard
from .transformer import TransformerBlock


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


def _resolve_model_dtype(config: dict, device: str) -> torch.dtype:
    desired = config.get("torch_dtype", torch.float32)
    if not isinstance(desired, torch.dtype) or not desired.is_floating_point:
        return torch.float32

    normalized = str(device).strip().lower()
    if normalized == "mps" and desired == torch.bfloat16:
        return torch.float16
    if normalized == "cpu" and desired == torch.float16:
        return torch.float32
    return desired


class Model(nn.Module):
    def __init__(
        self,
        config: dict,
        shard: Shard,
        use_tied: bool = False,
    ):
        super().__init__()
        self.config = config
        self.shard = shard
        self.device_name = get_backend_device("torch", default="cpu")
        assert self.device_name is not None
        self.inference_dtype = _resolve_model_dtype(self.config, self.device_name)

        print(f"loading shard: {shard}")

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.config["vocab_size"],
            embedding_dim=self.config["embed_dim"],
            padding_idx=self.config.get("pad_token_id"),
        )

        self.norm = _rms_norm(
            self.config["embed_dim"],
            eps=self.config["norm_eps"]
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(self.config, layer_idx=layer_idx)
                for layer_idx in range(self.shard.start_layer, self.shard.end_layer)
            ]
        )

        # output == lm_head
        lm_head_bias = bool(self.config.get("lm_head_bias", False))
        self.output = nn.Linear(
            self.config["embed_dim"],
            self.config["vocab_size"],
            bias=lm_head_bias,
        )
        self.to(device=self.device_name, dtype=self.inference_dtype)
        if use_tied:
            self.output.weight = self.embed_tokens.weight

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "kv_cache", None) is not None:
                attn.kv_cache.clear()

    def _resolve_shard_window(self, shard: Shard | None) -> tuple[int, int, bool]:
        requested = shard or self.shard
        loaded_start = int(getattr(self.shard, "start_layer", 0) or 0)
        loaded_end = int(getattr(self.shard, "end_layer", loaded_start + len(self.layers)) or (loaded_start + len(self.layers)))
        requested_start = int(getattr(requested, "start_layer", loaded_start) or loaded_start)
        requested_end = int(getattr(requested, "end_layer", loaded_end) or loaded_end)
        total_layers = int(getattr(requested, "total_layers", getattr(self.shard, "total_layers", loaded_end + 1)) or (loaded_end + 1))

        # Older shard payloads could advertise the virtual lm_head stage as the
        # exclusive end bound. Normalize that back to the loaded transformer
        # range so runtime slicing still works.
        if requested_end == total_layers and loaded_end == total_layers - 1:
            requested_end = loaded_end

        if requested_start < loaded_start or requested_end > loaded_end:
            raise ValueError(
                f"Requested shard {requested_start}:{requested_end} is outside loaded range {loaded_start}:{loaded_end}"
            )

        local_start = requested_start - loaded_start
        local_end = requested_end - loaded_start
        is_final = total_layers > 0 and requested_end >= total_layers - 1
        return local_start, local_end, is_final

    def run_shard(
        self,
        x: torch.Tensor | None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        shard: Shard | None = None,
        start_pos: int | None = None,
    ) -> torch.Tensor:
        del start_pos  # torch attention derives decode position from position_ids.
        if hidden_state is None:
            if x is None:
                raise ValueError("input tensor is required when hidden_state is not provided")
            x = self.embed_tokens(x.long())
        else:
            x = hidden_state.to(device=self.device_name, dtype=self.inference_dtype)

        local_start, local_end, is_final = self._resolve_shard_window(shard)
        for layer in self.layers[local_start:local_end]:
            x = layer(x, attention_mask, position_ids)

        if is_final:
            x = self.norm(x)
            x = self.output(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run_shard(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            hidden_state=hidden_state,
            shard=self.shard,
        )
