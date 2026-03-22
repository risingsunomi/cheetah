import math
from typing import Optional, Tuple

import tinygrad as tg

# From https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py
# From https://github.com/BatSmacker84/exo/blob/fbec1d2b10ccf3a804294c0742b69d508f7b5fb8/exo/inference/tinygrad/models/llama.py
def complex_multi(A: tg.Tensor, C: tg.Tensor, D: tg.Tensor) -> tg.Tensor:
  """
  real and imaginary complex multiplication
  """
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*C - b*D
  co = a*D + b*C
  return ro.cat(co, dim=-1)

class RotaryPositionalEmbedding:
    """
    A class for Rotary Position Embedding (ROPE) used in LLMs.
    Supports scaling and non-scaling ROPE
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
        dtype = tg.dtypes.half
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.is_scaling = is_scaling
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.scale_factor = scale_factor
        self.cache = None
        self.dtype = dtype
        self.cache_built = False

        # precompute theta
        self.theta = 1.0 / (self.base ** (
            tg.Tensor.arange(
                0,
                self.dim,
                2
            )[: (self.dim//2)] / self.dim
        ))
    
        if self.is_scaling:
            self.theta = self.apply_scaling(self.theta)

    def build_rope_cache(self, device) -> tg.Tensor:
        seq_idx = tg.Tensor.arange(self.max_seq_len * 2, device=device).float()
        self.freqs = (seq_idx.unsqueeze(1) * self.theta.unsqueeze(0))

        self.cache = tg.Tensor.stack([
           self.freqs.cos(),
           self.freqs.sin()
        ], dim=-1).reshape(
            1,
            self.max_seq_len * 2,
            1,
            self.dim // 2,
            2
        )

        self.cache_built = True

    def apply_scaling(
        self,
        freqs: tg.Tensor
    ) -> tg.Tensor:
        """
        Applies frequency scaling (e.g., YaRN-like) to the input frequency vector.
        Returns a tensor of the same shape as `freqs`.
        """
        # guard defaults if optional values are None
        low = 1.0 if (self.low_freq_factor is None) else float(self.low_freq_factor)
        high = 1.0 if (self.high_freq_factor is None) else float(self.high_freq_factor)
        # if old_context_len is not provided, don't change frequencies with this term
        if self.old_context_len is None or self.scale_factor is None:
            m = 1.0
        else:
            m = float(self.old_context_len) / float(self.max_seq_len)
            m = m ** (1.0 / float(self.scale_factor))

        # make a copy to avoid in-place on shared tensors
        out = (freqs * m).contiguous()
        # split low/high frequency bands roughly in half of the freqs (i.e., D/4 boundary for D/2 freqs)
        split_idx = self.dim // 4
        if split_idx > 0:
            out[:split_idx] = out[:split_idx] * low
            out[split_idx:] = out[split_idx:] * high
        else:
            out = out * high
        return out

    def __call__(
        self,
        q: tg.Tensor, # [B, S, Hq, D]
        k: tg.Tensor, # [B, S, Hk, D]
        position_ids: Optional[tg.Tensor] = None
    ) -> Tuple[tg.Tensor, tg.Tensor]:
        if not self.cache_built:
            self.build_rope_cache(q.device)

        """
        Returns rotated (q,k). Handles position_ids as [S], [B,S], or None.
        """
        D = q.shape[-1]
        Dk = k.shape[-1]
        assert D % 2 == 0 and Dk == D, "q/k head dims must match and be even"

        # reshape into complex pairs
        q_pairs = q.reshape(*q.shape[0:-1], -1, 2).float()
        k_pairs = k.reshape(*k.shape[0:-1], -1, 2).float()

        # select cos/sin for the provided positions from cache
        cs = self.cache[:, position_ids.cast(tg.dtypes.default_int), :, :, :]

        # split into cos/sin and apply rotation via complex multiply
        C, D_ = cs[..., 0:1], cs[..., 1:2]                      # [1, S, 1, D/2, 1]
        q_out = complex_multi(q_pairs, C, D_)                   # [B, S, Hq, D/2, 2]
        k_out = complex_multi(k_pairs, C, D_)                   # [B, S, Hk, D/2, 2]

        # flatten back to last dim and cast to original dtype
        return q_out.flatten(3).cast(q.dtype), k_out.flatten(3).cast(k.dtype)
