cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tensor_socket_test)

# Set libtorch path relative to root
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../libs/libtorch")

# PyTrorch C++ API (LibTorch)
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/../../libs/libtorch")
find_package(Torch REQUIRED)

# JSON library
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

# Build test server
add_executable(server tensor_server.cpp)
target_link_libraries(server PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
set_property(TARGET server PROPERTY CXX_STANDARD 17)
