#include "cache.h"

KVCacheImpl::KVCacheImpl(int64_t batch_size, int64_t max_seq_len, int64_t num_kv_heads, int64_t head_dim, torch::Dtype dtype) : batch_size(batch_size)
{
    auto cache_shape = std::vector<int64_t>{batch_size, num_kv_heads, max_seq_len, head_dim};

    k_cache = register_buffer("k_cache", torch::zeros(cache_shape, dtype));
    v_cache = register_buffer("v_cache", torch::zeros(cache_shape, dtype));
    cache_pos = register_buffer("cache_pos", torch::arange(0, max_seq_len));
}

void KVCacheImpl::reset()
{
    k_cache.zero_();
    v_cache.zero_();
    cache_pos -= size();
}

int64_t KVCacheImpl::size() const
{
    return cache_pos[0].item<int64_t>();
}


// Update the cache with new key and value tensors
std::tuple<torch::Tensor, torch::Tensor> KVCacheImpl::update(
    const torch::Tensor& k_val,
    const torch::Tensor& v_val
)
{
    // k_val, v_val: [B, H, S, D]
    batch_size = k_val.size(0);
    int64_t seq_len = k_val.size(2);

    if (batch_size > k_cache.size(0)) {
        throw std::runtime_error("Batch size exceeds cache capacity");
    }

    if (size() + seq_len <= k_cache.size(2)) {
        throw std::runtime_error("Sequence length exceeds cache capacity");
    }

    k_out = k_cache;
    v_out = v_cache;

    // Update cache position
    auto pos_slice = cache_pos.slice(0, 0, seq_len);
    
    // Update the cache with new key and value tensors
    k_out.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), pos_slice},
        k_val
    );
    
    v_out.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), pos_slice},
        v_val
    );

    cache_pos += seq_len;

    return std::make_tuple(k_out, v_out);

}

// RoPE Cache Manager
RopeCacheManager& RopeCacheManager::get() {
  static RopeCacheManager instance;
  return instance;
}

void RopeCacheManager::store(const std::string& key, const torch::Tensor& tensor) {
  cache[key] = tensor;
}

torch::Tensor RopeCacheManager::get(const std::string& key) {
  if (!contains(key)) {
    throw std::runtime_error("Cache miss: key not found -> " + key);
  }
  return cache.at(key);
}

bool RopeCacheManager::contains(const std::string& key) const {
  return cache.find(key) != cache.end();
}
