from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# From tinygrad/rope.py, adapted to torch.
def complex_multi(a: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    real = a[..., 0:1]
    imag = a[..., 1:2]
    ro = real * c - imag * d
    co = real * d + imag * c
    return torch.cat((ro, co), dim=-1)


class RotaryPositionalEmbedding:
    """
    Rotary Position Embedding with optional scaling.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 100000.0,
        is_scaling: bool = False,
        scale_factor: Optional[float] = None,
        low_freq_factor: Optional[float] = None,
        high_freq_factor: Optional[float] = None,
        old_context_len: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        mode: str = "interleaved",
        rope_type: str = "default",
        rope_truncate: bool = True,
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.is_scaling = is_scaling
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.scale_factor = scale_factor
        self.dtype = dtype
        self.mode = str(mode)
        self.rope_type = str(rope_type)
        self.rope_truncate = bool(rope_truncate)
        self.attention_scaling = 1.0

        self.cache: torch.Tensor | None = None
        self.cache_built = False

        theta_idx = torch.arange(0, self.dim, 2, dtype=torch.float32)[: self.dim // 2]
        self.theta = 1.0 / (self.base ** (theta_idx / self.dim))
        if self.is_scaling:
            if self.rope_type == "yarn":
                self.theta = self._apply_yarn_scaling(self.theta)
            else:
                self.theta = self.apply_scaling(self.theta)

    def build_rope_cache(self, device: torch.device) -> None:
        seq_idx = torch.arange(self.max_seq_len * 2, dtype=torch.float32, device=device)
        freqs = seq_idx.unsqueeze(1) * self.theta.to(device=device).unsqueeze(0)
        cache = torch.stack(
            (
                freqs.cos() * float(self.attention_scaling),
                freqs.sin() * float(self.attention_scaling),
            ),
            dim=-1,
        )
        self.cache = cache.reshape(1, self.max_seq_len * 2, 1, self.dim // 2, 2)
        self.cache_built = True

    def _apply_yarn_scaling(self, inv_freq_default: torch.Tensor) -> torch.Tensor:
        factor = float(self.scale_factor or 1.0)
        if factor <= 0.0:
            factor = 1.0

        def get_mscale(scale: float) -> float:
            if scale <= 1.0:
                return 1.0
            return 0.1 * math.log(scale) + 1.0

        self.attention_scaling = get_mscale(factor)

        beta_fast = float(self.low_freq_factor if self.low_freq_factor is not None else 32.0)
        beta_slow = float(self.high_freq_factor if self.high_freq_factor is not None else 1.0)
        original_max = float(self.old_context_len or self.max_seq_len)

        def find_correction_dim(num_rotations: float) -> float:
            return (self.dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(self.base))

        def find_correction_range(low_rot: float, high_rot: float) -> tuple[float, float]:
            low = find_correction_dim(low_rot)
            high = find_correction_dim(high_rot)
            if self.rope_truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0.0), min(high, float(self.dim - 1))

        def linear_ramp_factor(min_v: float, max_v: float, dim_half: int) -> torch.Tensor:
            if min_v == max_v:
                max_v += 0.001
            linear = (torch.arange(dim_half, dtype=torch.float32) - min_v) / (max_v - min_v)
            return torch.clamp(linear, 0.0, 1.0)

        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = find_correction_range(beta_fast, beta_slow)
        extrapolation_factor = 1.0 - linear_ramp_factor(low, high, self.dim // 2)
        return (
            inv_freq_interpolation * (1.0 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor
        )

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        low = 1.0 if self.low_freq_factor is None else float(self.low_freq_factor)
        high = 1.0 if self.high_freq_factor is None else float(self.high_freq_factor)

        if self.old_context_len is None or self.scale_factor is None:
            m = 1.0
        else:
            m = float(self.old_context_len) / float(self.max_seq_len)
            m = m ** (1.0 / float(self.scale_factor))

        out = freqs * m
        split_idx = self.dim // 4
        if split_idx > 0:
            out[:split_idx] = out[:split_idx] * low
            out[split_idx:] = out[split_idx:] * high
        else:
            out = out * high
        return out

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cache_built or self.cache is None or self.cache.device != q.device:
            self.build_rope_cache(q.device)

        head_dim = q.shape[-1]
        if head_dim % 2 != 0 or k.shape[-1] != head_dim:
            raise ValueError("q/k head dims must match and be even")

        if position_ids is None:
            position_ids = torch.arange(q.shape[1], device=q.device)
        elif position_ids.ndim == 2:
            position_ids = position_ids[0]

        index = position_ids.to(dtype=torch.long, device=q.device)
        cs = self.cache[:, index, :, :, :]

        c = cs[..., 0:1]
        d = cs[..., 1:2]
        if self.mode == "half":
            c_half = c.squeeze(-1)
            d_half = d.squeeze(-1)

            qf, qs = torch.chunk(q.float(), 2, dim=-1)
            kf, ks = torch.chunk(k.float(), 2, dim=-1)

            q_out = torch.cat((qf * c_half - qs * d_half, qs * c_half + qf * d_half), dim=-1)
            k_out = torch.cat((kf * c_half - ks * d_half, ks * c_half + kf * d_half), dim=-1)
            return q_out.to(dtype=q.dtype), k_out.to(dtype=k.dtype)

        q_pairs = q.reshape(*q.shape[:-1], -1, 2).float()
        k_pairs = k.reshape(*k.shape[:-1], -1, 2).float()
        q_out = complex_multi(q_pairs, c, d)
        k_out = complex_multi(k_pairs, c, d)

        return q_out.flatten(3).to(dtype=q.dtype), k_out.flatten(3).to(dtype=k.dtype)
