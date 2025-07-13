// KV Caching system based off of torchtune KVCache
#ifndef CACHE_H
#define CACHE_H

#include <torch/torch.h>
#include <unordered_map>
#include <string>

class KVCache : public torch::nn::Module {
    public:
        KVCache(int batch_size, int max_seq_len, int num_kv_heads, int head_dim, torch::Dtype dtype);
        void reset();
        int size() const;
        std::tuple<torch::Tensor, torch::Tensor> update(
            const torch::Tensor& k_val,
            const torch::Tensor& v_val
        );

        torch::Tensor k_cache;
        torch::Tensor v_cache;

    private:
        std::vector<int> cache_shape;
        int batch_size;
        torch::Tensor cache_pos;
        torch::Tensor k_out;
        torch::Tensor v_out;
};

class RopeCacheManager {
public:
  static RopeCacheManager& get();

  void store(const std::string& key, const torch::Tensor& tensor);
  torch::Tensor get(const std::string& key);
  bool contains(const std::string& key) const;

private:
  RopeCacheManager() = default;
  std::unordered_map<std::string, torch::Tensor> cache;
};

#endif // CACHE_H