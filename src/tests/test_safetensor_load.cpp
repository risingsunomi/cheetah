#include "../utils/safetensors_loader.h"
#include <iostream>

int main() {
  std::string path = "/home/t0kenl1mit/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/eb49081324edb2ff14f848ce16393c067c6f4976/model.safetensors";

  try {
    SafeTensorsLoader loader(path);
    auto tensors = loader.getTensors();
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}