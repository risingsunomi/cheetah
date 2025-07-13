cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tensor_socket_test)
include(FetchContent)

# # Set libtorch path relative to root
# set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../libs/libtorch")

# # PyTrorch C++ API (LibTorch)
# set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/../../libs/libtorch")
# find_package(Torch REQUIRED)

# JSON library
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# Build test server
# add_executable(server tensor_server.cpp)
add_executable(test_load_config
    test_load_model_config.cpp
    ../utils/model_config.cpp
)

# target_link_libraries(server PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
# set_property(TARGET server PROPERTY CXX_STANDARD 17)
target_link_libraries(test_load_config PRIVATE nlohmann_json::nlohmann_json)
set_property(TARGET test_load_config PROPERTY CXX_STANDARD 17)