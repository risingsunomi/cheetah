#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <netinet/in.h>

// Project includes
#include "../utils/model_config.h"
#include "../utils/shard.h"
#include "../general_mha_model.h"

using json = nlohmann::json;

ssize_t recv_all(int sock, void *buf, size_t len) {
  size_t received = 0;
  while (received < len) {
    ssize_t r = recv(sock, (char *)buf + received, len - received, 0);
    if (r <= 0) return r;
    received += r;
  }
  return received;
}

void send_all(int sock, const void *buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    ssize_t s = send(sock, (char *)buf + sent, len - sent, 0);
    if (s <= 0) throw std::runtime_error("send failed");
    sent += s;
  }
}

torch::Tensor recv_tensor_view(
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

void send_tensor(int sock, const torch::Tensor &tensor) {
  std::vector<int> shape_vec(tensor.sizes().begin(), tensor.sizes().end());
  
  json header = {
    {"command", "response"},
    {"dtype", "bfloat16"},
    {"shape", shape_vec}
  };

  std::string header_str = header.dump();
  uint32_t header_len = htonl(header_str.size());
  
  send_all(sock, &header_len, 4);
  send_all(sock, header_str.data(), header_str.size());
  send_all(sock, tensor.data_ptr(), tensor.nbytes());
}

int main() {
  const char *socket_path = "/run/cheetah_infra";
  unlink(socket_path);
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
  bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
  listen(server_fd, 1);

  std::cout << "Server listening on " << socket_path << std::endl;
  int client_fd = accept(server_fd, nullptr, nullptr);

  // Receive and parse header
  uint32_t header_len_be;
  recv_all(client_fd, &header_len_be, 4);
  uint32_t header_len = ntohl(header_len_be);
  std::vector<char> header_buf(header_len);
  recv_all(client_fd, header_buf.data(), header_len);
  json header = json::parse(header_buf.begin(), header_buf.end());

  std::string model_id = header["model"];
  std::string config_path = header["config_path"];
  int layer_start = header["layer_start"];
  int layer_end = header["layer_end"];
  int layer_total = header["layer_total"]
  std::string dtype = header["dtype"];
  std::vector<int> shape_input = header["shape_input"];
  std::vector<int> shape_mask = header["shape_mask"];

  // Load shard
  Shard shard(model_id, layer_start, layer_end
  // Load model config
  std::cout << "Loading model config @ " << config_path << std::endl;
  ModelConfig config(config_path);
  int total_layers = config.num_layers;

  std::cout << "Layer range: " << layer_start << " to " << layer_end << " / " << total_layers << std::endl;
  std::cout << "Hidden dim: " << config.embed_dim << " | Heads: " << config.num_attention_heads << std::endl;

  // Receive tensors
  torch::ScalarType scalar = (dtype == "int64") ? torch::kInt64 : throw std::runtime_error("Unsupported dtype");
  auto scalar_size = torch::elementSize(scalar);
  auto input_prod = torch::prod(torch::tensor(shape_input));
  auto mask_prod = torch::prod(torch::tensor(shape_mask));
  size_t total_bytes = scalar_size * (
    input_prod.item<int>() + mask_prod.item<int>());

  std::vector<char> buffer(total_bytes);
  recv_all(client_fd, buffer.data(), total_bytes);
  size_t offset = 0;
  const char *ptr = buffer.data();
  
  torch::Tensor input_ids = recv_tensor_view(
    ptr,
    offset,
    shape_input,
    scalar
  );
  
  torch::Tensor attention_mask = recv_tensor_view(
    ptr,
    offset,
    shape_mask,
    scalar
  );

  std::cout << "Input IDs shape: " << input_ids.sizes() << "\n";
  std::cout << "Attention mask shape: " << attention_mask.sizes() << "\n";

  // Run GeneralMHA inference
  GeneralMHAModel model(config, layer_start, layer_end, total_layers);
  model.eval();
  torch::InferenceMode guard;

  auto logits = model.forward(input_ids, attention_mask);
  std::cout << "Logits shape: " << logits.sizes() << "\n";

  // Send logits back
  send_tensor(client_fd, logits);

  // Clean up
  close(client_fd);
  close(server_fd);
  unlink(socket_path);
  return 0;
}
