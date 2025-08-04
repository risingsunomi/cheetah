cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cheetah)
include(FetchContent)

# PyTrorch C++ API (LibTorch)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)

# JSON library
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(main
  main.cpp
  src/attention.cpp
  src/mlp.cpp
  src/rms.cpp
  src/rope.cpp
  src/transformers.cpp
  src/general_mha_model.cpp
  
  src/utils/cache.cpp
  src/utils/shard.cpp
  src/utils/model_config.cpp
  src/utils/safetensors_loader.cpp
  src/utils/helpers.cpp
  src/utils/sessionmgmt.cpp
)


target_link_libraries(main "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
set_property(TARGET main PROPERTY CXX_STANDARD 17)