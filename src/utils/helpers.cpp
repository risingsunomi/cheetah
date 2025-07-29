#include "helpers.h"

bool Helpers::is_low_memory(
  const std::string os = "linux",
  size_t threshold_mb = 20000
) {

  if (os == "linux") {
    struct sysinfo info;
    if (sysinfo(&info) != 0)
    return false;

    unsigned long total_ram_mb = info.totalram * info.mem_unit / (1024 * 1024);
    return total_ram_mb < threshold_mb;
  }

  return false;
}

bool Helpers::llama_detect(std::string model_name_) {
  static const std::regex rgx_llama(
    "llama",
    std::regex_constants::icase
  );
  
  return std::regex_search(model_name_, rgx_llama);
}

bool Helpers::model_ver_detect(
  const std::string model_name_,
  const std::string target_version_
) {
  static const std::regex rgx_ver(
    target_version_,
    std::regex_constants::icase
  );

  return std::regex_search(model_name_, rgx_ver);
}

ssize_t Helpers::recv_all(int sock, void *buf, size_t len) {
  size_t received = 0;
  while (received < len) {
    ssize_t r = recv(sock, (char *)buf + received, len - received, 0);
    if (r <= 0) return r;
    received += r;
  }
  return received;
}

void Helpers::send_all(int sock, const void *buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    ssize_t s = send(sock, (char *)buf + sent, len - sent, 0);
    if (s <= 0) throw std::runtime_error("send failed");
    sent += s;
  }
}

torch::Tensor Helpers::recv_tensor_view(
  const char *&buffer_ptr,
  size_t &offset,
  const std::vector<int> &shape,
  torch::ScalarType dtype
) {
  size_t numel = 1;
  for (auto s : shape) numel *= s;
  size_t byte_size = numel * torch::elementSize(dtype);
  std::vector<int64_t> shape64(shape.begin(), shape.end());
  torch::Tensor tensor = torch::from_blob(
    (void *)(buffer_ptr + offset),
    shape64,
    torch::TensorOptions().dtype(dtype)
  ).clone();
  offset += byte_size;
  return tensor;
}

void Helpers::send_tensor(int sock, const torch::Tensor &tensor, std::string dtype) {
  std::vector<int> shape_vec(tensor.sizes().begin(), tensor.sizes().end());
  
  nlohmann::json header = {
    {"command", "response"},
    {"dtype", dtype},
    {"shape", shape_vec}
  };

  std::string header_str = header.dump();
  uint32_t header_len = htonl(header_str.size());
  
  send_all(sock, &header_len, 4);
  send_all(sock, header_str.data(), header_str.size());
  send_all(sock, tensor.data_ptr(), tensor.nbytes());
}

torch::Dtype Helpers::dtype_from_string(const std::string& name) {
  static const std::unordered_map<std::string, torch::Dtype> map = {
    {"float16", torch::kFloat16},
    {"bfloat16", torch::kBFloat16},
    {"float32", torch::kFloat32},
    {"float64", torch::kFloat64},
    {"int8", torch::kInt8},
    {"int16", torch::kInt16},
    {"int32", torch::kInt32},
    {"int64", torch::kInt64},
    {"uint8", torch::kUInt8},
    {"bool", torch::kBool},
    {"complex64", torch::kComplexFloat},
    {"complex128", torch::kComplexDouble},
    {"half", torch::kFloat16},
    {"float", torch::kFloat32},
    {"double", torch::kFloat64},
    {"long", torch::kInt64},
    {"int", torch::kInt32},
    {"short", torch::kInt16},
    {"byte", torch::kUInt8},
    {"char", torch::kInt8},
    {"qint8", torch::kQInt8},
    {"quint8", torch::kQUInt8},
    {"qint32", torch::kQInt32},
    {"float8_e5m2", torch::kFloat8_e5m2},
    {"float8_e4m3fn", torch::kFloat8_e4m3fn},
    {"float8_e4m3fnuz", torch::kFloat8_e4m3fnuz},
    {"float8_e5m2fnuz", torch::kFloat8_e5m2fnuz}
  };

  auto it = map.find(name);
  if (it == map.end()) {
    throw std::invalid_argument("Unsupported dtype string: " + name);
  }
  return it->second;
}