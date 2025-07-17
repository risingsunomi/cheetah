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

class SafeTensorsLoader {
public:
  explicit SafeTensorsLoader(const std::string& filename);
  ~SafeTensorsLoader();
  std::vector<std::string> keys() const;
  std::unordered_map<std::string, torch::Tensor> getTensors();

private:
  void parseHeader();
  int fd;
  size_t file_size;
  uint64_t header_len;
  void* map_ptr;
  uint8_t* data_ptr;
  std::unordered_map<std::string, nlohmann::json> json_map;
  std::unordered_map<std::string, torch::Tensor> safetensors;
};

#endif // SAFETENSORS_LOADER_H