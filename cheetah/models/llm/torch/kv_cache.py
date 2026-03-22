from __future__ import annotations

from typing import Any, Tuple

import torch


class KVCache:
    def __init__(
        self,
        max_cache_len: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ):
        del max_seq_len
        self.max_cache_len = int(max_cache_len)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.device = self._resolve_device(device)
        self.dtype = dtype or torch.float32
        self.cache_kv = torch.zeros(
            (2, batch_size, self.max_cache_len, self.num_kv_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.cache_pos = 0

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            normalized = device.strip().lower()
            if normalized in {"cuda", "gpu"} and torch.cuda.is_available():
                return torch.device("cuda")
            if normalized in {"mps", "metal"} and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device("cpu")

    def clear(self) -> None:
        self.cache_kv = torch.zeros_like(self.cache_kv)
        self.cache_pos = 0

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        bsz, seq_len, kv_heads, dim = xk.shape
        if kv_heads != self.num_kv_heads or dim != self.head_dim:
            raise ValueError("KV shapes mismatch cache settings")
        if self.cache_pos + seq_len > self.max_cache_len:
            raise ValueError(
                f"seq len {seq_len} exceeds max cache len {self.max_cache_len}"
            )

        xk = xk.to(device=self.cache_kv.device, dtype=self.cache_kv.dtype)
        xv = xv.to(device=self.cache_kv.device, dtype=self.cache_kv.dtype)
        kv_stack = torch.stack((xk, xv), dim=0)
        self.cache_kv[:, :bsz, self.cache_pos : self.cache_pos + seq_len, :, :] = kv_stack
        self.cache_pos += seq_len

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_pos == 0:
            batch_size = self.cache_kv.shape[1]
            empty_shape = (batch_size, 0, self.num_kv_heads, self.head_dim)
            return (
                torch.zeros(empty_shape, dtype=self.cache_kv.dtype, device=self.cache_kv.device),
                torch.zeros(empty_shape, dtype=self.cache_kv.dtype, device=self.cache_kv.device),
            )

        keys = self.cache_kv[0, :, : self.cache_pos, :, :]
        values = self.cache_kv[1, :, : self.cache_pos, :, :]
        return keys, values

    def serialize(self) -> dict[str, Any]:
        return {
            "max_len": self.max_cache_len,
            "cache_pos": self.cache_pos,
            "cache_kv": self.cache_kv.detach().cpu().numpy(),
        }
