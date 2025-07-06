cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cheetah)

set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libs/libtorch")

find_package(Torch REQUIRED)

add_executable(main
  main.cpp
  models/general_mha.cpp
  models/rope.cpp
  models/cache.cpp
)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)