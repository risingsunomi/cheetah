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
#include <csignal>
#include <thread>
#include <fcntl.h>
#include <chrono>

#include "src/general_mha_model.h"
#include "src/utils/shard.h"
#include "src/utils/model_config.h"
#include "src/utils/helpers.h"

using json = nlohmann::json;

const char *socket_path = "/tmp/cheetah_infra";
Helpers util_helper;

void cleanup(int) {
  unlink(socket_path);
  std::exit(0);
}

void handle_client(int client_fd) {
  std::cout << "handle_client" << std::endl;
  try {
    uint32_t header_len_be;
    ssize_t recv_header_len = util_helper.recv_all(client_fd, &header_len_be, 4);
    std::cout << std::to_string(recv_header_len) << std::endl;
    uint32_t header_len = ntohl(header_len_be);
    std::cout << "header_len " << std::to_string(header_len) << std::endl;

    std::vector<char> header_buf(header_len);
    util_helper.recv_all(client_fd, header_buf.data(), header_len);
    json header = json::parse(header_buf.begin(), header_buf.end());

    std::string node_id = header["node_id"];
    std::string model_id = header["model"];
    std::string config_path = header["config_path"];
    std::vector<std::string> safetensors_path = header["safetensors_path"];
    int layer_start = header["layer_start"];
    int layer_end = header["layer_end"];
    int layer_total = header["layer_total"];
    std::string dtype = header["dtype"];
    std::vector<int> shape_input = header["shape_input"];
    std::vector<int> shape_mask = header["shape_mask"];

    std::cout << "Received request from node: " << node_id << std::endl;
    Shard shard(model_id, layer_start, layer_end, layer_total);
    std::cout << "Loading model config @ " << config_path << std::endl;
    ModelConfig config(config_path);
    int total_layers = config.num_layers;
    std::cout << "Layer range: " << layer_start << " to " << layer_end << " / " << total_layers << std::endl;

    torch::ScalarType scalar;
    if (dtype == "int64") scalar = torch::kInt64;
    else if (dtype == "int32") scalar = torch::kInt32;
    else throw std::runtime_error("Unsupported dtype");

    auto scalar_size = torch::elementSize(scalar);
    auto input_prod = torch::prod(torch::tensor(shape_input));
    auto mask_prod = torch::prod(torch::tensor(shape_mask));
    size_t tensor_total_bytes = scalar_size * (input_prod.item<int>() + mask_prod.item<int>());

    std::vector<char> buffer(tensor_total_bytes);
    util_helper.recv_all(client_fd, buffer.data(), tensor_total_bytes);

    size_t offset = 0;
    const char *ptr = buffer.data();

    torch::Tensor input_ids = util_helper.recv_tensor_view(ptr, offset, shape_input, scalar);
    torch::Tensor attention_mask = util_helper.recv_tensor_view(ptr, offset, shape_mask, scalar);

    std::cout << "Input IDs shape: " << input_ids.sizes() << std::endl;
    std::cout << "Attention mask shape: " << attention_mask.sizes() << std::endl;

    std::cout << "Loading GeneralMHAModel" << std::endl;
    auto model = GeneralMHAModel(
        shard,
        config,
        safetensors_path,
        config.use_cache);

    // TODO: Perform model inference or response handling here
  } catch (const std::exception &e) {
    std::cerr << "Error while handling client: " << e.what() << std::endl;
  }

  close(client_fd);
}

int main() {
  std::cout << "Starting Cheetah Service..." << std::endl;
  std::signal(SIGINT, cleanup);

  unlink(socket_path);
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd == -1) {
    perror("socket");
    return 1;
  }

  // Set socket to non-blocking mode
  int flags = fcntl(server_fd, F_GETFL, 0);
  fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

  if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
    perror("bind");
    return 1;
  }

  if (listen(server_fd, 128) == -1) {
    perror("listen");
    return 1;
  }

  std::cout << "Socket started @ " << socket_path << std::endl;

  while (true) {
    util_helper.print_waiting();
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      } else {
        perror("accept failed");
        break;
      }
    }

    std::thread(handle_client, client_fd).detach();
  }

  close(server_fd);
  return 0;
}
