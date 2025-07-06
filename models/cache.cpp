#include "cache.h"

KVCacheImpl::KVCacheImpl(int64_t batch_size, int64_t max_seq_len, int64_t num_kv_heads, int64_t head_dim, torch::Dtype dtype) : batch_size(batch_size)
{
    std::vector<int64_t> cache_shape = std::vector<int64_t>{batch_size, num_kv_heads, max_seq_len, head_dim};

    k_cache = register_buffer("k_cache", torch::zeros(cache_shape, dtype));
    v_cache = register_buffer("v_cache", torch::zeros(cache_shape, dtype));
    cache_pos = register_buffer("cache_pos", torch::arange(0, max_seq_len));
}