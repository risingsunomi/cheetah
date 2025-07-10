#ifndef SHARD_H
#define SHARD_H

#include <string>
#include <algorithm>

class Shard {
public:
    Shard(const std::string& model_id, int start_layer, int end_layer, int n_layers);
    std::string model_id;
    int start_layer;
    int end_layer;
    int n_layers;

    bool is_first_layer() const;
    bool is_last_layer() const;
    int get_layer_count() const;
    bool overlaps(const Shard& other) const;
};

#endif // SHARD_H