from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .attention import _prepare_attention_mask
from .kv_cache import KVCache


class MambaAttention(nn.Module):
    def __init__(self, config: dict, layer_idx: int | None = None):
        super().__init__()
        del layer_idx
        self.hidden_size = int(config["embed_dim"])
        self.num_heads = int(config["num_heads"])
        self.num_key_value_heads = int(config["num_kv_heads"])
        self.head_dim = int(config.get("attention_head_dim", config.get("head_dim", self.hidden_size // self.num_heads)))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = float(config.get("attn_dropout", 0.0))
        self.max_seq_len = int(config.get("max_seq_len", 2048))
        bias = bool(config.get("attn_bias", config.get("qkv_bias", False)))
        self.kv_cache: KVCache | None = None

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        del position_ids
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if self.kv_cache is None:
            self.kv_cache = KVCache(
                self.max_seq_len,
                batch_size,
                self.num_key_value_heads,
                self.max_seq_len,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        self.kv_cache.update(key_states, value_states)
        key_states, value_states = self.kv_cache.get()

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.num_key_value_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        if attention_mask is not None and attention_mask.ndim == 4:
            attn_mask = attention_mask[:, :, :, : key_states.shape[-2]].to(
                device=query_states.device,
                dtype=query_states.dtype,
            )
        else:
            attn_mask = _prepare_attention_mask(
                attention_mask,
                query_states,
                key_len=key_states.shape[-2],
                is_causal=True,
            )
        if query_states.device.type == "cuda" and attn_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        is_causal = bool(attn_mask is None and seq_len > 1)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
