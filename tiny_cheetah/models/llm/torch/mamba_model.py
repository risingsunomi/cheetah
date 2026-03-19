from __future__ import annotations

import torch
from torch import nn

from tiny_cheetah.models.llm.backend import get_backend_device
from ...shard import Shard
from .mamba_block import MambaHybridBlock
from .mamba_mixer import MambaStateCache


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


class MambaBackbone(nn.Module):
    def __init__(self, config: dict, shard: Shard):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=int(config["vocab_size"]),
            embedding_dim=int(config["embed_dim"]),
            padding_idx=config.get("pad_token_id"),
        )
        self.layers = nn.ModuleList(
            [
                MambaHybridBlock(config, layer_idx=layer_idx)
                for layer_idx in range(shard.start_layer, shard.end_layer)
            ]
        )
        self.norm_f = _rms_norm(
            int(config["embed_dim"]),
            eps=float(config.get("layer_norm_epsilon", config.get("norm_eps", 1e-6))),
        )


class MambaModel(nn.Module):
    def __init__(self, config: dict, shard: Shard, use_tied: bool = False):
        super().__init__()
        self.config = config
        self.shard = shard
        self.device_name = get_backend_device("torch", default="cpu")
        assert self.device_name is not None
        self.inference_dtype = _resolve_model_dtype(self.config, self.device_name)
        self.backbone = MambaBackbone(config, shard)
        self.lm_head = nn.Linear(
            int(config["embed_dim"]),
            int(config["vocab_size"]),
            bias=bool(config.get("lm_head_bias", False)),
        )
        if use_tied:
            self.lm_head.weight = self.backbone.embeddings.weight
        self._mamba_cache: MambaStateCache | None = None
        self._mamba_cache_position: int = 0
        self.to(device=self.device_name, dtype=self.inference_dtype)

    def _ensure_mamba_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._mamba_cache is None or self._mamba_cache.batch_size != batch_size:
            self._mamba_cache = MambaStateCache(
                self.config,
                batch_size,
                dtype=dtype,
                device=device,
            )
            self._mamba_cache_position = 0

    def reset_kv_cache(self) -> None:
        for layer in self.backbone.layers:
            mixer = getattr(layer, "mixer", None)
            if mixer is not None and getattr(mixer, "kv_cache", None) is not None:
                mixer.kv_cache = None
        self._mamba_cache = None
        self._mamba_cache_position = 0

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        dtype = input_tensor.dtype
        device = input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = int(cache_position[-1].item()) + 1

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) & attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                    padding_mask,
                    min_dtype,
                )

        return causal_mask

    @staticmethod
    def _update_mamba_mask(
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        if cache_position[0] > 0:
            return None
        if attention_mask is not None and torch.all(attention_mask == 1):
            return None
        return attention_mask

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_state is None:
            hidden_states = self.backbone.embeddings(x.long())
        else:
            hidden_states = hidden_state

        batch_size, seq_len, _ = hidden_states.shape
        self._ensure_mamba_cache(batch_size, hidden_states.device, hidden_states.dtype)
        cache_position = None
        if any(block.block_type == "mamba" for block in self.backbone.layers):
            cache_position = torch.arange(
                self._mamba_cache_position,
                self._mamba_cache_position + seq_len,
                device=hidden_states.device,
                dtype=torch.long,
            )
        if cache_position is None:
            cache_position = torch.arange(
                self._mamba_cache_position,
                self._mamba_cache_position + seq_len,
                device=hidden_states.device,
                dtype=torch.long,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position)
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        for layer in self.backbone.layers:
            if layer.block_type == "mamba":
                layer_mask = mamba_mask
            elif layer.block_type == "attention":
                layer_mask = causal_mask
            else:
                layer_mask = None
            hidden_states = layer(
                hidden_states,
                layer_mask,
                position_ids,
                mamba_cache=self._mamba_cache,
                cache_position=cache_position,
            )

        self._mamba_cache_position += seq_len

        if self.shard.end_layer == self.shard.total_layers - 1:
            hidden_states = self.backbone.norm_f(hidden_states)
            hidden_states = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        return hidden_states
