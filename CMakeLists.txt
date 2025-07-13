cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cheetah)

# PyTrorch C++ API (LibTorch)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)

# JSON library
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

add_executable(main
  main.cpp
  src/attention.cpp
  src/mlp.cpp
  src/rms.cpp
  src/rope.cpp
  src/general_mha_model.cpp
  src/transformers.cpp

  src/utils/cache.cpp
  src/utils/shard.cpp
)


target_link_libraries(main "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
set_property(TARGET main PROPERTY CXX_STANDARD 17)