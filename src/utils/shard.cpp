#include "shard.h"

Shard::Shard(
    const std::string model_id_,
    int& start_layer_,
    int& end_layer_,
    int& n_layers_)
    : model_id(model_id_),
    start_layer(start_layer_),
    end_layer(end_layer_),
    n_layers(n_layers_) {}

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