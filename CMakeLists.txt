cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES "89")
project(cook-cuda CUDA)
enable_language(C)
enable_language(CXX)

set(CMAKE_CUDA_STANDARD 14)
include_directories(include)

add_executable(main src/main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main PRIVATE rt cuda)