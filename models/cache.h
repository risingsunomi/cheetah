// KV Caching system based off of torchtune KVCache
#include <torch/torch.h>
#include <unordered_map>
#include <string>

class KVCacheImpl : public torch::nn::Module {
    public:
        KVCacheImpl(int64_t batch_size, int64_t max_seq_len, int64_t num_kv_heads, int64_t head_dim, torch::Dtype dtype);
        void reset();
        int64_t size() const;
        std::tuple<torch::Tensor, torch::Tensor> update(
            const torch::Tensor& k_val,
            const torch::Tensor& v_val
        );

    private:
        std::vector<int> cache_shape;
        int64_t batch_size;
        torch::Tensor k_cache;
        torch::Tensor v_cache;
        torch::Tensor cache_pos;
        torch::Tensor k_out;
        torch::Tensor v_out;
};
TORCH_MODULE(KVCache);

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