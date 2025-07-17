#include "rope.h"

RotaryEmbedding::RotaryEmbedding(
  int dim_,
  int max_seq_len_,
  float base_,
  c10::optional<std::string> cache_key_
) : dim(dim_),
  max_seq_len(max_seq_len_),
  base(base_),
  cache_key(cache_key_) {
  rope_init();
}

void RotaryEmbedding::rope_init() {
  auto indices = torch::arange(0, dim, 2, torch::kFloat32).slice(0, 0, dim / 2);
  theta = 1.0f / torch::pow(static_cast<float>(base), indices / static_cast<float>(dim));

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

void RotaryEmbedding::build_rope_cache(int max_seq_len_) {
  auto seq_idx = torch::arange(max_seq_len_, theta.options());
  auto idx_theta = torch::einsum("i,j->ij", {seq_idx, theta}).to(torch::kFloat32);

  auto cos_vals = torch::cos(idx_theta);
  auto sin_vals = torch::sin(idx_theta);
  cache = torch::stack({cos_vals, sin_vals}, -1);  // [max_seq_len_, dim/2, 2]
}

torch::Tensor RotaryEmbedding::forward(
    const torch::Tensor& x_,
    const torch::Tensor& input_pos_
) {
    int seq_len = x_.size(1);
    torch::Tensor rope_cache;
    if (!input_pos_.defined()) {
        rope_cache = cache.index({torch::indexing::Slice(0, seq_len)});
    } else {
        rope_cache = cache.index_select(0, input_pos_.view(-1));
    }
    
    auto xshaped = x_.to(torch::kFloat32).reshape({x_.size(0), x_.size(1), x_.size(2), -1, 2});
    rope_cache = rope_cache.view({-1, xshaped.size(1), 1, xshaped.size(3), 2});

    auto x_out = torch::stack({
        xshaped.index({"...", 0}) * rope_cache.index({"...", 0}) - 
        xshaped.index({"...", 1}) * rope_cache.index({"...", 1}),
        xshaped.index({"...", 1}) * rope_cache.index({"...", 0}) + 
        xshaped.index({"...", 0}) * rope_cache.index({"...", 1})
    }, -1);

    x_out = x_out.flatten(3);
    return x_out.to(x_.dtype());
}