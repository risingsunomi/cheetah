#include "safetensors_loader.h"

using json = nlohmann::json;

SafeTensorsLoader::SafeTensorsLoader(const std::string& filename) {
  fd = ::open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    ::close(fd);
    throw std::runtime_error("Failed to stat file: " + filename);
  }
  file_size = st.st_size;

  map_ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map_ptr == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("mmap failed for file: " + filename);
  }

  size_t headerLen = *reinterpret_cast<uint64_t*>(map_ptr);
  parseHeader();
  data_ptr = reinterpret_cast<uint8_t*>(map_ptr) + 8 + headerLen;
}

SafeTensorsLoader::~SafeTensorsLoader() {
  munmap(map_ptr, file_size);
  close(fd);
}

void SafeTensorsLoader::parseHeader() {
  uint64_t headerLen = *reinterpret_cast<uint64_t*>(map_ptr);
  char* headerStart = reinterpret_cast<char*>(map_ptr) + 8;

  std::string headerJson(headerStart, headerStart + headerLen);
  json j = json::parse(headerJson);

  for (auto& item : j.items()) {
    json_map[item.key()] = item.value();
  }
}

std::vector<std::string> SafeTensorsLoader::keys() const {
  std::vector<std::string> names;
  names.reserve(json_map.size());
  for (const auto& kv : json_map) {
    if(kv.first == "__metadata__") continue;
    names.push_back(kv.first);
  }
  return names;
}

std::unordered_map<std::string, torch::Tensor> SafeTensorsLoader::getTensors() {
    auto ks = keys();
    for(auto& k : ks){
        auto name = k;

        auto it = json_map.find(name);
        if (it == json_map.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        auto& meta = it->second;

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

        safetensors.emplace(name, tensor);

    }
    return safetensors;
}