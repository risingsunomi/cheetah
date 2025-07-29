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

void Helpers::print_waiting() {
  static int state = 0;
  const char cursor[] = {'|', '/', '-', '\\'};
  std::cout << "\r Waiting" << cursor[state++ % 4] << std::flush;
}