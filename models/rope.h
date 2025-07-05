#include <torch/torch.h>

class RotaryEmbedding {
    public:
        RotaryEmbedding(int64_t dim, int64_t max_seq_len, float base = 10000.0f);
        torch::Tensor apply_rotary(const torch::Tensor& x);

    private:
        int64_t dim;
        int64_t max_seq_len;
        float base;

        torch::Tensor cos_cache;
        torch::Tensor sin_cache;
};