cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES "89")
project(cook-cuda CUDA)
enable_language(C)
enable_language(CXX)
enable_language(CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0146 OLD)
include(FindCUDA)
cuda_select_nvcc_arch_flags(ARCH_FLAGS 7.5 8.9)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
message(STATUS "CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")

include_directories(include)

find_package(fmt CONFIG REQUIRED)

file(GLOB_RECURSE SRCS
        src/sample/*.cu)

add_executable(main src/main.cu ${SRCS})
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main PRIVATE fmt::fmt rt cuda)