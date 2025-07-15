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
        int dim_,
        int max_seq_len_,
        float base_,
        c10::optional<std::string> cache_key_ = c10::nullopt);

    torch::Tensor forward(
        const torch::Tensor &x_,
        const torch::Tensor &input_pos_);

private:
    void rope_init();
    void build_rope_cache(int max_seq_len_);

    int dim;
    int max_seq_len;
    int base;
    bool is_cache_built = false;
    torch::Tensor theta;
    torch::Tensor cache;
    c10::optional<std::string> cache_key;
};

#endif