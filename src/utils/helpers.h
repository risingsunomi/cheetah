// Helper functions
#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <string>
#include <sys/sysinfo.h>
#include <regex>
#include <sys/socket.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <netinet/in.h>
#include <chrono>
#include <thread>

class Helpers {
  public:
  bool is_low_memory(const std::string os_, size_t threshold_mb_);
  bool llama_detect(std::string model_name_);
  bool model_ver_detect(
    const std::string model_name_,
    const std::string target_version_
  );
  ssize_t recv_all(int sock, void *buf, size_t len);
  void send_all(int sock, const void *buf, size_t len);
  torch::Tensor recv_tensor_view(
    const char *&buffer_ptr,
    size_t &offset,
    const std::vector<int> &shape,
    torch::ScalarType dtype
  );
  void send_tensor(
    int sock,
    const torch::Tensor &tensor,
    const std::string dtype = "float32"
  );
  void print_waiting();

  const std::string os = "linux";
  size_t threshold_mb = 20000;
};

#endif