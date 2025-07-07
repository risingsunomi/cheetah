#include "rope.h"

RotaryEmbedding::RotaryEmbedding(
  int dim,
  int max_seq_len,
  int base,
  bool use_scaling,
  int scale_factor,
  int low_freq_factor,
  int high_freq_factor,
  int old_context_len,
  std::optional<std::string> cache_key
) : dim(dim), max_seq_len(max_seq_len), base(base), use_scaling(use_scaling),
    scale_factor(scale_factor), low_freq_factor(low_freq_factor),
    high_freq_factor(high_freq_factor), old_context_len(old_context_len),
    cache_key(cache_key) {
  rope_init();
}

void RotaryEmbedding::rope_init() {
  auto freqs = 1.0 / torch::pow(
    torch::tensor(base, torch::kFloat32),
    torch::arange(0, dim, 2, torch::kFloat32) / dim
  );

  if (freqs.device().is_meta()) return;

  theta = use_scaling
    ? apply_scaling(
        freqs,
        scale_factor,
        low_freq_factor,
        high_freq_factor,
        old_context_len
      )
    : freqs;

  if (cache_key && RopeCacheManager::get().contains(*cache_key)) {
    cache = RopeCacheManager::get().get(*cache_key);
  } else {
    build_rope_cache(max_seq_len);
    if (cache_key) {
      RopeCacheManager::get().store(*cache_key, cache);
    }
  }

  is_cache_built = true;
}

torch::Tensor RotaryEmbedding::apply_scaling(
  const torch::Tensor& freqs,
  int scale_factor,
  int low_freq_factor,
  int high_freq_factor,
  int old_context_len
) {
  float low_freq_wavelen = static_cast<float>(old_context_len) / low_freq_factor;
  float high_freq_wavelen = static_cast<float>(old_context_len) / high_freq_factor;

  std::vector<float> new_freqs;
  auto freqs_acc = freqs.accessor<float, 1>();

  for (int i = 0; i < freqs.size(0); ++i) {
    float freq = freqs_acc[i];
    float wavelen = 2 * M_PI / freq;

    if (wavelen < high_freq_wavelen) {
      new_freqs.push_back(freq);
    } else if (wavelen > low_freq_wavelen) {
      new_freqs.push_back(freq / scale_factor);
    } else {
      float smooth = (old_context_len / wavelen - low_freq_factor) /
                     (high_freq_factor - low_freq_factor);
      new_freqs.push_back((1 - smooth) * freq / scale_factor + smooth * freq);
    }
  }

  return torch::tensor(new_freqs, freqs.options());
}

void RotaryEmbedding::build_rope_cache(int max_seq_len) {
  auto seq_idx = torch::arange(max_seq_len, theta.options());
  auto idx_theta = torch::einsum("i,j->ij", {seq_idx, theta}).to(torch::kFloat32);

  auto cos_vals = torch::cos(idx_theta);
  auto sin_vals = torch::sin(idx_theta);
  cache = torch::stack({cos_vals, sin_vals}, -1);  // [max_seq_len, dim/2, 2]
}

torch::Tensor RotaryEmbedding::apply(const torch::Tensor& x, const torch::Tensor& input_pos) {
  if (!is_cache_built) {
    TORCH_CHECK(false, "RoPE cache not built. Call rope_init() first.");
  }

  int64_t b = x.size(0);
  int64_t s = x.size(1);
  int64_t nh = x.size(2);
  int64_t hd = x.size(3);

  torch::Tensor rope_cache = input_pos.defined()
    ? cache.index_select(0, input_pos.view(-1))
    : cache.index({torch::indexing::Slice(0, s)});

  auto xshaped = x.reshape({b, s, nh, hd / 2, 2}).to(torch::kFloat32);
  rope_cache = rope_cache.view({1, s, 1, hd / 2, 2});

  auto out = torch::stack({
    xshaped.select(-1, 0) * rope_cache.select(-1, 0) - xshaped.select(-1, 1) * rope_cache.select(-1, 1),
    xshaped.select(-1, 1) * rope_cache.select(-1, 0) + xshaped.select(-1, 0) * rope_cache.select(-1, 1)
  }, -1);

  return out.flatten(3).to(x.dtype());
}
