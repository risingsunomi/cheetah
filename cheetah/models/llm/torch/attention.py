from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .kv_cache import KVCache
from .rope import RotaryPositionalEmbedding


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


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    query: torch.Tensor,
    *,
    key_len: int,
    is_causal: bool,
) -> Optional[torch.Tensor]:
    if attention_mask is None and not is_causal:
        return None

    bool_mask: Optional[torch.Tensor] = None
    additive_mask: Optional[torch.Tensor] = None

    if attention_mask is not None:
        mask = attention_mask.to(device=query.device)

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if mask.ndim == 2:
            mask = mask[:, None, None, :key_len]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :key_len]
        else:
            mask = mask[..., :key_len]

        if mask.dtype == torch.bool:
            bool_mask = mask
        elif mask.dtype.is_floating_point:
            additive_mask = mask.to(dtype=query.dtype)
        else:
            bool_mask = mask > 0

    if is_causal:
        query_len = query.shape[-2]
        offset = max(key_len - query_len, 0)
        query_idx = torch.arange(offset, offset + query_len, device=query.device)[:, None]
        key_idx = torch.arange(key_len, device=query.device)[None, :]
        causal_mask = (key_idx <= query_idx).view(1, 1, query_len, key_len)
        bool_mask = causal_mask if bool_mask is None else (bool_mask & causal_mask)

    if bool_mask is not None:
        zero = torch.zeros((), dtype=query.dtype, device=query.device)
        neg_inf = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
        bool_mask = torch.where(bool_mask, zero, neg_inf)
        additive_mask = bool_mask if additive_mask is None else (additive_mask + bool_mask)

    if additive_mask is None:
        return None
    return additive_mask.to(dtype=query.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        config: dict,
        is_causal: Optional[bool] = False,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.kv_cache: KVCache | None = None

        self.qkv_bias = bool(config.get("qkv_bias", config.get("attn_bias", False)))
        self.o_proj_bias = bool(config.get("o_proj_bias", config.get("attn_bias", False)))
        self.attn_dropout = config["attn_dropout"]
        self.is_causal = bool(is_causal)
        self.model_type = str(config.get("model_type", "")).lower()
        self.use_attention_sinks = self.model_type == "gpt_oss"
        self.sliding_window: int | None = None
        if self.model_type == "gpt_oss":
            layer_types = config.get("layer_types") or []
            if (
                isinstance(layer_types, list)
                and layer_idx is not None
                and 0 <= layer_idx < len(layer_types)
                and str(layer_types[layer_idx]) == "sliding_attention"
            ):
                sw = int(config.get("sliding_window", 0) or 0)
                self.sliding_window = sw if sw > 0 else None

        if config["rope_scaling"] is not None:
            self.pos_embeddings = RotaryPositionalEmbedding(
                self.head_dim,
                self.max_seq_len,
                config["rope_theta"],
                is_scaling=True,
                scale_factor=config["rope_scaling_factor"],
                low_freq_factor=config["rope_low_freq_factor"],
                high_freq_factor=config["rope_high_freq_factor"],
                old_context_len=config["rope_original_max_pos_embeddings"],
                mode="half" if self.model_type == "gpt_oss" else "interleaved",
                rope_type=config.get("rope_type", "default"),
                rope_truncate=bool(config.get("rope_truncate", True)),
            )
        else:
            self.pos_embeddings = RotaryPositionalEmbedding(
                self.head_dim,
                self.max_seq_len,
                config["rope_theta"],
                mode="half" if self.model_type == "gpt_oss" else "interleaved",
                rope_type=config.get("rope_type", "default"),
                rope_truncate=bool(config.get("rope_truncate", True)),
            )

        self.k_norm = _rms_norm(self.head_dim) if config["qk_norm"] else None
        self.q_norm = _rms_norm(self.head_dim) if config["qk_norm"] else None

        self.q_proj = nn.Linear(
            self.embed_dim,
            self.num_heads * self.head_dim,
            bias=self.qkv_bias,
        )
        self.k_proj = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=self.qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=self.qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=self.o_proj_bias,
        )
        if self.use_attention_sinks:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.register_parameter("sinks", None)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_proj(x)

        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.reshape(batch_size, seq_len, self.num_kv_heads * q_per_kv, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        elif position_ids.ndim == 2:
            position_ids = position_ids[0]

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self.pos_embeddings(q, k, position_ids)

        if self.training:
            if self.kv_cache is not None and self.kv_cache.cache_pos != 0:
                self.kv_cache.clear()
        else:
            if self.kv_cache is None:
                self.kv_cache = KVCache(
                    self.max_seq_len,
                    batch_size,
                    self.num_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                    dtype=x.dtype,
                    device=x.device,
                )

            self.kv_cache.update(k, v)
            k, v = self.kv_cache.get()

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            q_per_kv = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        if self.use_attention_sinks:
            bsz, heads, q_len, _ = q.shape
            k_len = k.shape[-2]

            attn_logits = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)

            offset = k_len - q_len
            query_idx = torch.arange(offset, offset + q_len, device=q.device)[:, None]
            key_idx = torch.arange(k_len, device=q.device)[None, :]
            causal_mask = key_idx <= query_idx

            if self.sliding_window is not None:
                lower = query_idx - (self.sliding_window - 1)
                causal_mask = causal_mask & (key_idx >= lower)

            attn_logits = attn_logits.masked_fill(
                ~causal_mask.view(1, 1, q_len, k_len),
                torch.finfo(attn_logits.dtype).min,
            )

            if attention_mask is not None:
                if attention_mask.ndim == 1:
                    key_mask = attention_mask.unsqueeze(0)
                elif attention_mask.ndim == 2:
                    key_mask = attention_mask
                else:
                    key_mask = None
                if key_mask is not None:
                    key_mask = key_mask[:, -k_len:].to(device=q.device) > 0
                    attn_logits = attn_logits.masked_fill(
                        ~key_mask[:, None, None, :],
                        torch.finfo(attn_logits.dtype).min,
                    )
                else:
                    attn_mask = _prepare_attention_mask(attention_mask, q)
                    if attn_mask is not None:
                        attn_logits = attn_logits + attn_mask

            sink_logits = self.sinks.reshape(1, -1, 1, 1).to(device=q.device, dtype=attn_logits.dtype)
            sink_logits = sink_logits.expand(bsz, -1, q_len, -1)
            combined = torch.cat((attn_logits, sink_logits), dim=-1)
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probs = torch.softmax(combined.to(torch.float32), dim=-1).to(dtype=attn_logits.dtype)
            scores = probs[..., :-1]
            attn_out = torch.matmul(scores, v)
        else:
            attn_mask = _prepare_attention_mask(
                attention_mask,
                q,
                key_len=k.shape[-2],
                is_causal=self.is_causal,
            )
            use_causal = self.is_causal if attn_mask is None else False

            dropout_p = self.attn_dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
            )

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_out)
