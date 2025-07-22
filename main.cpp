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

    // to do - handle hidden state
    std::string node_id = header["node_id"];
    std::string model_id = header["model"];
    std::string model_path = header["model_path"];
    int layer_start = header["layer_start"];
    int layer_end = header["layer_end"];
    int layer_total = header["layer_total"];
    std::string dtype = header["dtype"];
    std::vector<int> input_id_shape = header["input_id_shape"];
    std::vector<int> mask_shape = header["mask_shape"];
    std::vector<int> input_pos_shape = header["input_pos_shape"];

    std::cout << "Received request from node: " << node_id << std::endl;
    
    Shard shard(
      model_id,
      layer_start,
      layer_end,
      layer_total
    );

    std::cout << "Loading model config @ " << model_path << std::endl;
    ModelConfig config(model_path);
    
    std::cout << "Layer range: " << layer_start << " to " << layer_end << " / " << layer_total << std::endl;

    // recieve tensors
    torch::ScalarType scalar;
    if (dtype == "int64") scalar = torch::kInt64;
    else if (dtype == "int32") scalar = torch::kInt32;
    else throw std::runtime_error("Unsupported dtype");

    auto scalar_size = torch::elementSize(scalar);
    auto input_id_prod = torch::prod(torch::tensor(input_id_shape));
    auto mask_prod = torch::prod(torch::tensor(mask_shape));
    auto input_pos_prod = torch::prod(torch::tensor(input_pos_shape));
    size_t tensor_total_bytes = scalar_size * (
      input_id_prod.item<int>() +
      mask_prod.item<int>() +
      input_pos_prod.item<int>()
    );

    std::vector<char> buffer(tensor_total_bytes);
    util_helper.recv_all(
      client_fd,
      buffer.data(),
      tensor_total_bytes
    );

    size_t offset = 0;
    const char *ptr = buffer.data();
    torch::Tensor input_ids = util_helper.recv_tensor_view(
      ptr,
      offset,
      input_id_shape,
      scalar
    );
    torch::Tensor attention_mask = util_helper.recv_tensor_view(
      ptr,
      offset,
      mask_shape,
      scalar
    );
    torch::Tensor input_pos = util_helper.recv_tensor_view(
      ptr,
      offset,
      input_pos_shape,
      scalar
    );

    std::cout << "Input IDs shape: " << input_ids.sizes() << std::endl;
    std::cout << "Attention mask shape: " << attention_mask.sizes() << std::endl;

    std::cout << "Loading GeneralMHAModel" << std::endl;
    auto model = GeneralMHAModel(
      shard,
      config,
      model_path,
      config.use_cache);
    
    model.eval();

    // infer forward
    torch::Tensor model_out = model.forward(
      input_ids,
      attention_mask,
      input_pos,
      c10::nullopt);

    // return tensor
    util_helper.send_tensor(client_fd, model_out);
    close(client_fd);
    
  } catch (const std::exception &e) {
    std::cerr << "Error while handling client: " << e.what() << std::endl;
  }
}

void show_logo() {
  std::cout << R"(

░░      ░░░  ░░░░  ░░        ░░        ░░        ░░░      ░░░  ░░░░  ░
▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒
▓  ▓▓▓▓▓▓▓▓        ▓▓      ▓▓▓▓      ▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓        ▓
█  ████  ██  ████  ██  ████████  ███████████  █████        ██  ████  █
██      ███  ████  ██        ██        █████  █████  ████  ██  ████  █
                                                                                                                                      
  )" << std::endl;
}

int main() {
  show_logo();
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
  std::cout << "Waiting for input data..." << std::endl;

  while (true) {
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
