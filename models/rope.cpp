#include "rope.h"
#include <cmath>

RotaryEmbedding::RotaryEmbedding(int64_t dim, int64_t max_seq_len, float base)
    : dim(dim), max_seq_len(max_seq_len), base(base) {
  auto inv_freq = torch::pow(base, -torch::arange(0, dim, 2, torch::kFloat32) / dim);
  auto t = torch::arange(max_seq_len, torch::kFloat32);
  auto freqs = torch::outer(t, inv_freq);  // [max_seq_len, dim/2]

  cos_cache = torch::cos(freqs).unsqueeze(1);  // [max_seq_len, 1, dim/2]
  sin_cache = torch::sin(freqs).unsqueeze(1);  // [max_seq_len, 1, dim/2]
}

torch::Tensor RotaryEmbedding::apply_rotary(const torch::Tensor& x) {
  // x: [B, H, T, D]
  auto B = x.size(0), H = x.size(1), T = x.size(2), D = x.size(3);

  auto x1 = x.slice(-1, 0, D, 2);
  auto x2 = x.slice(-1, 1, D, 2);

  auto cos = cos_cache.index({torch::indexing::Slice(0, T)}).permute({1, 0, 2});
  auto sin = sin_cache.index({torch::indexing::Slice(0, T)}).permute({1, 0, 2});

  cos = cos.expand({B, T, D / 2});
  sin = sin.expand({B, T, D / 2});

  auto rot_x = torch::cat({x1 * cos - x2 * sin, x2 * cos + x1 * sin}, -1);
  return rot_x.view({B, H, T, D});
}
