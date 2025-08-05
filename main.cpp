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
#include "src/utils/sessionmgmt.h"

using json = nlohmann::json;

const char *socket_path = "/tmp/cheetah_infra";
Helpers util_helper;

void cleanup(int) {
  unlink(socket_path);
  std::exit(0);
}

void handle_client(int client_fd, SessionManagement &session_mgmt) {
  try {
    uint32_t header_len_be;
    ssize_t recv_header_len = util_helper.recv_all(client_fd, &header_len_be, 4);
    uint32_t header_len = ntohl(header_len_be);

    std::vector<char> header_buf(header_len);
    util_helper.recv_all(client_fd, header_buf.data(), header_len);
    std::cout << "Recieved header" << std::endl;
    json header = json::parse(header_buf.begin(), header_buf.end());

    // to do - handle hidden state
    std::string session_id = header["session_id"];
    std::string node_id = header["node_id"];
    std::cout << "Session ID: " << session_id << std::endl;
    std::cout << "Node ID: " << node_id << std::endl;

    std::string model_id = header["model"];
    int layer_start = header["layer_start"];
    int layer_end = header["layer_end"];
    int layer_total = header["layer_total"];
  
    Shard shard(
      model_id,
      layer_start,
      layer_end,
      layer_total
    );

    std::cout << "Shard: " << shard.model_id << " from layer "
      << shard.start_layer << " to " << shard.end_layer
      << " / " << shard.n_layers << std::endl;
    
    std::string model_path = header["model_path"];
    ModelConfig config(model_path + "/config.json");
    std::cout << "Model Config: " << config.config_path << std::endl;
    
    torch::Tensor model_out;
    torch::Tensor input_ids;
    torch::Tensor attention_mask;
    torch::Tensor input_pos;
    torch::Tensor hidden_state;

    if(header["has_hidden_state"]) {
      std::string hidden_state_dtype = header["hidden_state_dtype"];
      std::vector<int> hidden_state_shape = header["hidde_state_shape"];

      size_t tensor_total_bytes =
        torch::elementSize(util_helper.dtype_from_string(hidden_state_dtype)) *
        torch::prod(torch::tensor(hidden_state_shape)).item<int>();
      std::vector<char> buffer(tensor_total_bytes);
      util_helper.recv_all(client_fd, buffer.data(), tensor_total_bytes);
      size_t offset = 0;
      const char *ptr = buffer.data();
      torch::Tensor hidden_state = util_helper.recv_tensor_view(
        ptr,
        offset,
        hidden_state_shape,
        util_helper.dtype_from_string(hidden_state_dtype)
      );

      std::cout << "Hidden state shape: " << hidden_state.sizes() << std::endl;
      std::cout << "Hidden state dtype: " << hidden_state.dtype() << std::endl;

      // infer forward
      std::cout << "Running model forward with hidden state..." << std::endl;
    } else {
      torch::ScalarType dtype_input_id = util_helper.dtype_from_string(
        header["dtype_input_ids"]
      );
      torch::ScalarType dtype_mask = util_helper.dtype_from_string(
        header["dtype_mask"]
      );
      torch::ScalarType dtype_input_pos = util_helper.dtype_from_string(
        header["dtype_input_pos"]
      );
      std::vector<int> input_id_shape = header["input_ids_shape"];
      std::vector<int> mask_shape = header["mask_shape"];
      std::vector<int> input_pos_shape = header["input_pos_shape"];

      // recieve tensors
      size_t tensor_total_bytes =
      torch::elementSize(dtype_input_id) *
      torch::prod(torch::tensor(input_id_shape)).item<int>() +
      torch::elementSize(dtype_mask) *
      torch::prod(torch::tensor(mask_shape)).item<int>() +
      torch::elementSize(dtype_input_pos) *
      torch::prod(torch::tensor(input_pos_shape)).item<int>();

      std::vector<char> buffer(tensor_total_bytes);
      util_helper.recv_all(
        client_fd,
        buffer.data(),
        tensor_total_bytes
      );

      size_t offset = 0;
      const char *ptr = buffer.data();
      input_ids = util_helper.recv_tensor_view(
        ptr,
        offset,
        input_id_shape,
        dtype_input_id
      );
      
      attention_mask = util_helper.recv_tensor_view(
        ptr,
        offset,
        mask_shape,
        dtype_mask
      );
      attention_mask = attention_mask.to(torch::kBool);

      input_pos = util_helper.recv_tensor_view(
        ptr,
        offset,
        input_pos_shape,
        dtype_input_pos
      );

      std::cout << "Input IDs shape: " << input_ids.sizes() << std::endl;
      std::cout << "Attention mask shape: " << attention_mask.sizes() << std::endl;
      std::cout << "Input POS shape: " << input_pos.sizes() << std::endl; 
    }

    // check if has a session model
    if(session_mgmt.has_session(session_id, node_id)) {
      std::cout << "Found session, reusing model" << std::endl;
      auto smodel = session_mgmt.get_model(session_id, node_id);
      
      // infer forward
      std::cout << "Running model forward..." << std::endl;
      model_out = smodel->forward(
        input_ids,
        attention_mask,
        input_pos,
        c10::nullopt);
    } else {
      std::cout << "No session found, creating new model" << std::endl;
      auto smodel = std::make_shared<GeneralMHAModel>(
        shard,
        config,
        model_path,
        config.use_cache
      );
      session_mgmt.put(session_id, node_id, smodel);

      // infer forward
      std::cout << "Running model forward..." << std::endl;
      model_out = smodel->forward(
        input_ids,
        attention_mask,
        input_pos,
        c10::nullopt);
    }
      
    // return tensor
    std::cout << "Returning model tensor output of shape: " << model_out.sizes() << std::endl;
    util_helper.send_tensor(client_fd, model_out.to(torch::kFloat32));
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

  SessionManagement session_mgmt = SessionManagement();
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

    //std::thread(handle_client, client_fd, session_mgmt).detach();
    handle_client(client_fd, session_mgmt);
  }

  close(server_fd);
  return 0;
}
