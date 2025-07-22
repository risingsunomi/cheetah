#include "safetensors_loader.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

SafeTensorsLoader::SafeTensorsLoader(
  const std::string model_path_,
  Shard shard_
) : model_path(model_path_), shard(shard_) {}

SafeTensorsLoader::~SafeTensorsLoader() {
  munmap(map_ptr, file_size);
  close(fd);
}

std::string SafeTensorsLoader::searchIndex(
  const std::string index_path_,
  const std::string weight_name_
) {
  json json_index = json::parse(index_path_);

  for (auto& [key, shard_info] : json_index["weight_map"].items()) { 
    if(key == weight_name_) {
      return model_path + "/" + shard_info.get<std::string>();
    }
  }

  throw std::runtime_error(
    "Could not find safetensor " + 
    weight_name_ +
    " in index " +
    index_path_
  );

}

torch::Tensor SafeTensorsLoader::findWeight(
  const std::string weight_name_
) {
  fs::path path(model_path);
  if (fs::is_directory(path)) {
    auto safetensor_index = model_path +
      "/model.safetensors.index.json";

    if (fs::exists(safetensor_index)) {
      auto safetensor_path = searchIndex(
        safetensor_index,
        weight_name_
      );

      return loadWeight(
        safetensor_path,
        weight_name_
      );
    } else {
      for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".safetensors") {
          return loadWeight(
            entry.path().string(),
            weight_name_);
        }
      }
    }
  }
  
  throw std::runtime_error(
    "Could not find model directory " + model_path);
}

torch::Tensor SafeTensorsLoader::loadWeight(
  const std::string file_path_,
  const std::string weight_name_
) {
  fd = ::open(file_path_.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("Failed to open file: " + file_path_);
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    ::close(fd);
    throw std::runtime_error("Failed to stat file: " + file_path_);
  }
  file_size = st.st_size;

  map_ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map_ptr == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("mmap failed for file: " + file_path_);
  }

  // parse header
  size_t headerLen = *reinterpret_cast<uint64_t*>(map_ptr);
  char* headerStart = reinterpret_cast<char*>(map_ptr) + 8;

  std::string headerJson(headerStart, headerStart + headerLen);
  json j = json::parse(headerJson);

  for (auto& item : j.items()) {
    if(item.key() == weight_name_)
    json_map[item.key()] = item.value();
  }

  data_ptr = reinterpret_cast<uint8_t*>(map_ptr) + 8 + headerLen;
  
  for(const auto& kv : json_map){
    if(kv.first.find(weight_name_) != std::string::npos) {
      auto& meta = kv.second;

      std::string dtype = meta["dtype"].get<std::string>();
      std::vector<int64_t> shape;
      for (auto& d : meta["shape"]) {
          shape.push_back(d.get<int64_t>());
      }

      uint64_t start = meta["data_offsets"][0].get<uint64_t>();
      uint64_t end   = meta["data_offsets"][1].get<uint64_t>();
      size_t byteLen = end - start;

      torch::ScalarType scalarType;
      if (dtype == "F32") scalarType = torch::kFloat32;
      else if (dtype == "F16") scalarType = torch::kFloat16;
      else if (dtype == "I64") scalarType = torch::kInt64;
      else if (dtype == "I32") scalarType = torch::kInt32;
      else if (dtype == "BF16") scalarType = torch::kBFloat16;
      else throw std::runtime_error("Unsupported dtype: " + dtype);

      void *ptr = data_ptr + start;
      auto tensor = torch::from_blob(ptr, shape, torch::TensorOptions().dtype(scalarType)).clone();

      return tensor;
    }
  }

  throw std::runtime_error(
    "Could not load weight " + 
    weight_name_ +
    " from safetensor " +
    file_path_
  );
    
}