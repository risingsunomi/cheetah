#ifndef ROPE_H
#define ROPE_H

#include <torch/torch.h>
#include <optional>
#include <string>
#include "utils/cache.h"

class RotaryEmbedding
{
public:
    RotaryEmbedding(
        int dim,
        int max_seq_len = 4096,
        int base = 10000,
        bool use_scaling = true,
        int scale_factor = 8,
        int low_freq_factor = 1,
        int high_freq_factor = 4,
        int old_context_len = 8192,
        std::optional<std::string> cache_key = std::nullopt);

    torch::Tensor apply(const torch::Tensor &x, const torch::Tensor &input_pos);

private:
    void rope_init();
    void build_rope_cache(int max_seq_len);
    torch::Tensor apply_scaling(
        const torch::Tensor &freqs,
        int scale_factor,
        int low_freq_factor,
        int high_freq_factor,
        int old_context_len);

    int dim, max_seq_len, base;
    bool use_scaling;
    int scale_factor, low_freq_factor, high_freq_factor, old_context_len;
    bool is_cache_built = false;
    torch::Tensor theta, cache;
    std::optional<std::string> cache_key;
};

#endif // ROPE_H