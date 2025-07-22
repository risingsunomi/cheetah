#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

// PyTorch C++ safetensor loader
// ref: https://leetarxiv.substack.com/p/parsing-safetensors-file-format

#include <string>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include "shard.h"

class SafeTensorsLoader {
public:
  explicit SafeTensorsLoader(
    const std::string model_path_,
    Shard shard_
  );

  ~SafeTensorsLoader();
  torch::Tensor findWeight(const std::string weight_name_);
  

private:
  torch::Tensor loadWeight(
    const std::string filen_path_,
    const std::string weight_name_
  );

  std::string searchIndex(
    const std::string index_path_,
    const std::string weight_name_
  );
  
  int fd;
  size_t file_size;
  uint64_t header_len;
  void* map_ptr;
  uint8_t* data_ptr;
  std::string model_path;
  Shard shard;
  std::unordered_map<std::string, nlohmann::json> json_map;
};

#endif // SAFETENSORS_LOADER_H