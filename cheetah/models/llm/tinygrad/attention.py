from typing import Optional

import tinygrad as tg

from .kv_cache import KVCache
from .rope import RotaryPositionalEmbedding


def _prepare_attention_mask(
    attention_mask: Optional[tg.Tensor],
    query: tg.Tensor,
    *,
    key_len: int,
    is_causal: bool,
) -> Optional[tg.Tensor]:
    if attention_mask is None and not is_causal:
        return None

    bool_mask: Optional[tg.Tensor] = None
    additive_mask: Optional[tg.Tensor] = None

    if attention_mask is not None:
        mask = attention_mask.to(query.device)

        if len(mask.shape) == 1:
            mask = mask.reshape(1, mask.shape[0])

        if len(mask.shape) == 2:
            mask = mask[:, None, None, :key_len]
        elif len(mask.shape) == 3:
            mask = mask[:, None, :, :key_len]
        else:
            mask = mask[..., :key_len]

        if mask.dtype == tg.dtypes.bool:
            bool_mask = mask
        elif tg.dtypes.is_float(mask.dtype):
            additive_mask = mask
        else:
            bool_mask = mask > 0

    if is_causal:
        query_len = query.shape[-2]
        offset = max(key_len - query_len, 0)
        query_idx = tg.Tensor.arange(offset, offset + query_len, device=query.device).reshape(1, 1, query_len, 1)
        key_idx = tg.Tensor.arange(key_len, device=query.device).reshape(1, 1, 1, key_len)
        causal_mask = key_idx <= query_idx
        bool_mask = causal_mask if bool_mask is None else (bool_mask & causal_mask)

    if bool_mask is not None:
        bool_mask = bool_mask.where(0.0, -float("inf"))
        additive_mask = bool_mask if additive_mask is None else (additive_mask + bool_mask)

    return additive_mask

class MultiHeadAttention:
    def __init__(
        self,
        config: dict,
        is_causal: Optional[bool] = False,
    ):
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        # initialize internally; don't rely on config keys that may not exist
        self.kv_cache = None
        self.qkv_bias = bool(config.get("qkv_bias", config.get("attn_bias", False)))
        self.o_proj_bias = bool(config.get("o_proj_bias", config.get("attn_bias", False)))
        self.attn_dropout = config["attn_dropout"]
        self.is_causal = is_causal

        if config["rope_scaling"] is not None:
            self.pos_embeddings = RotaryPositionalEmbedding(
                self.head_dim,
                self.max_seq_len,
                config["rope_theta"],
                is_scaling=True,
                scale_factor=config["rope_scaling_factor"],
                low_freq_factor=config["rope_low_freq_factor"],
                high_freq_factor=config["rope_high_freq_factor"],
                old_context_len=config["rope_original_max_pos_embeddings"]
            )
        else:
            self.pos_embeddings = RotaryPositionalEmbedding(
                self.head_dim,
                self.max_seq_len,
                config["rope_theta"]
            )

        self.k_norm = tg.nn.RMSNorm(self.head_dim) if config["qk_norm"] else None
        self.q_norm = tg.nn.RMSNorm(self.head_dim) if config["qk_norm"] else None

        self.q_proj = tg.nn.Linear(
            self.embed_dim,
            self.num_heads * self.head_dim,
            bias=self.qkv_bias
        )
        self.k_proj = tg.nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=self.qkv_bias
        )
        self.v_proj = tg.nn.Linear(
            self.embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=self.qkv_bias
        )
        self.o_proj = tg.nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=self.o_proj_bias
        )

    def __call__(
        self,
        x: tg.Tensor,
        attention_mask: Optional[tg.Tensor] = None,
        position_ids: Optional[tg.Tensor] = None,
    ) -> tg.Tensor:
        """
        Written for purely self attention so missing cross attention
        methods
        """
        batch_size, x_seq_len, _ = x.shape

        # k,v shape [batch, embed_dim, num_kv_heads * head_dim]
        # q shape [batch, embed_dim, num_heads * head_dim]
        k,v,q = self.k_proj(x), self.v_proj(x), self.q_proj(x)
        
        # Q should have num_heads heads; K/V may have fewer (GQA)
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.reshape(batch_size, x_seq_len, self.num_kv_heads * q_per_kv, self.head_dim)
        k = k.reshape(batch_size, x_seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, x_seq_len, self.num_kv_heads, self.head_dim)

        if position_ids is None:
            position_ids = tg.Tensor.arange(x_seq_len, device=x.device)
        elif position_ids.ndim == 2:
            position_ids = position_ids[0]

        if self.q_norm:
            q = self.q_norm(q)

        if self.k_norm:
            k = self.k_norm(k)

        q, k = self.pos_embeddings(q, k, position_ids)

        # initialize cache with correct dtype/device lazily
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

        # Update KV cache using pre-transpose K/V: [B, S, Kv, D]
        self.kv_cache.update(k, v)
        k, v = self.kv_cache.get()  # [B, T, Kv, D]

        # transpose for SDPA
        # q: [B, H, S, D], k/v: [B, Kv, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        
        

        # Repeat Kv heads to match H (GQA)
        if self.num_kv_heads != self.num_heads:
            B, Kv, T, D = k.shape
            q_per_kv = self.num_heads // self.num_kv_heads
            try:
                k = k.reshape(B, Kv, 1, T, D).expand((B, Kv, q_per_kv, T, D)).flatten(1, 2)
                v = v.reshape(B, Kv, 1, T, D).expand((B, Kv, q_per_kv, T, D)).flatten(1, 2)
            except Exception:
                zeros_k = tg.Tensor.zeros((B, Kv, q_per_kv, T, D), device=k.device)
                zeros_v = tg.Tensor.zeros((B, Kv, q_per_kv, T, D), device=v.device)
                k = (k.reshape(B, Kv, 1, T, D) + zeros_k).flatten(1, 2)
                v = (v.reshape(B, Kv, 1, T, D) + zeros_v).flatten(1, 2)

        attn_mask = _prepare_attention_mask(
            attention_mask,
            q,
            key_len=k.shape[-2],
            is_causal=self.is_causal,
        )
        use_causal = self.is_causal if attn_mask is None else False
        attn_out = tg.Tensor.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout,
            is_causal=use_causal
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, x_seq_len, -1)
        return self.o_proj(attn_out)
