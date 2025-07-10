#include "../utils/shard.h"

Shard::Shard(const std::string& model_id, int start_layer, int end_layer, int n_layers)
    : model_id(model_id),
    start_layer(start_layer),
    end_layer(end_layer),
    n_layers(n_layers) {}

bool Shard::is_first_layer() const {
    return start_layer == 0;
}

bool Shard::is_last_layer() const {
    return end_layer == n_layers - 1;
}

int Shard::get_layer_count() const {
    return end_layer - start_layer + 1;
}

bool Shard::overlaps(const Shard& other) const {
    return model_id == other.model_id && std::max(start_layer, other.start_layer) <= std::min(end_layer, other.end_layer);
}