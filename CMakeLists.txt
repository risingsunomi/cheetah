cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cheetah)

set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libs/libtorch")

find_package(Torch REQUIRED)

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
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)