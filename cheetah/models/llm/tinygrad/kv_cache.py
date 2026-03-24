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
        """Reset logical cache length without reallocating the buffer."""
        self.cache_pos = 0

    def update_at(self, start_pos: int | tg.UOp, xk: tg.Tensor, xv: tg.Tensor) -> None:
        """
        Write into the cache at an explicit start position. TinyJit decode needs
        the position to stay a runtime input instead of captured Python state.
        """
        _, S, Kv, D = xk.shape
        assert Kv == self.num_kv_heads and D == self.head_dim, "KV shapes mismatch cache settings"
        assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"dtype mismatch: {xk.dtype=}, {xv.dtype=}, cache={self.cache_kv.dtype}"

        xk = xk.to(self.cache_kv.device).cast(self.cache_kv.dtype)
        xv = xv.to(self.cache_kv.device).cast(self.cache_kv.dtype)
        kv_stack = tg.Tensor.stack(xk, xv)
        self.cache_kv[:, :, start_pos:start_pos + S, :, :].assign(kv_stack).realize()  # type: ignore[index]
        if not isinstance(start_pos, tg.UOp):
            self.cache_pos = max(self.cache_pos, int(start_pos) + S)

    def update(self, xk: tg.Tensor, xv: tg.Tensor) -> None:
        """
        xk, xv: [B, S, Kv, D]
        Appends to cache_kv along time dimension using shrink+assign.
        """
        S = xk.shape[1]
        assert (self.cache_pos + S) <= self.max_cache_len, f"seq len {S} exceeds max cache len {self.max_cache_len}"
        self.update_at(self.cache_pos, xk, xv)

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
        return self.get_upto(self.cache_pos)

    def get_upto(self, end_pos: int | tg.UOp) -> Tuple[tg.Tensor, tg.Tensor]:
        """
        Return cache contents up to an explicit end position. This keeps the
        decode length symbolic under TinyJit.
        """
        keys = self.cache_kv[0, :, 0:end_pos, :, :]
        values = self.cache_kv[1, :, 0:end_pos, :, :]
        return keys, values

    def serialize(self) -> dict:
        return {
            "max_len": self.max_cache_len,
            "cache_pos": self.cache_pos,
            "cache_kv": self.cache_kv.numpy(),
        }
