from typing import Tuple
import tinygrad as tg

class KVCache:
    def __init__(
        self,
        max_cache_len: int,
        batch_size,
        num_kv_heads,
        max_seq_len,
        head_dim,
        *,
        dtype=None,
        device=None,
    ):
        # unified KV cache: [2, B, T_max, Kv, D]
        self.max_cache_len = max_cache_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache_kv = tg.Tensor.zeros(
            (2, batch_size, max_cache_len, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        ).contiguous().realize()
        self.cache_pos = 0

    def clear(self) -> None:
        """Reset cached keys/values without dropping allocated tensor."""
        self.cache_kv = tg.Tensor.zeros_like(self.cache_kv).contiguous().realize()
        self.cache_pos = 0

    def update(self, xk: tg.Tensor, xv: tg.Tensor) -> None:
        """
        xk, xv: [B, S, Kv, D]
        Appends to cache_kv along time dimension using shrink+assign.
        """
        B, S, Kv, D = xk.shape
        assert Kv == self.num_kv_heads and D == self.head_dim, "KV shapes mismatch cache settings"
        assert (self.cache_pos + S) <= self.max_cache_len, f"seq len {S} exceeds max cache len {self.max_cache_len}"

        # Ensure dtype matches
        assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"dtype mismatch: {xk.dtype=}, {xv.dtype=}, cache={self.cache_kv.dtype}"

        # stack K,V -> [2, B, S, Kv, D] and write into time slice
        xk = xk.to(self.cache_kv.device).cast(self.cache_kv.dtype)
        xv = xv.to(self.cache_kv.device).cast(self.cache_kv.dtype)
        kv_stack = tg.Tensor.stack(xk, xv)
        self.cache_kv.shrink((None, None, (self.cache_pos, self.cache_pos + S), None, None)).assign(kv_stack).realize()
        self.cache_pos += S

    def get(self) -> Tuple[tg.Tensor, tg.Tensor]:
        """
        Returns all cached keys/values up to cache_pos.
        shapes: [B, T, Kv, D]
        """
        if self.cache_pos == 0:
            # empty cache; return empty tensors for safety
            B = self.cache_kv.shape[1]
            return (
                tg.Tensor.zeros((B, 0, self.num_kv_heads, self.head_dim), device=self.cache_kv.device, dtype=self.cache_kv.dtype),
                tg.Tensor.zeros((B, 0, self.num_kv_heads, self.head_dim), device=self.cache_kv.device, dtype=self.cache_kv.dtype),
            )
        keys = self.cache_kv[0].shrink((None, (0, self.cache_pos), None, None))
        values = self.cache_kv[1].shrink((None, (0, self.cache_pos), None, None))
        return keys, values

    def serialize(self) -> dict:
        return {
            "max_len": self.max_cache_len,
            "cache_pos": self.cache_pos,
            "cache_kv": self.cache_kv.numpy(),
        }
