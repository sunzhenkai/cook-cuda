#include "hello_world.cuh"
#include "stdio.h"
#include "utils/utils.cuh"

__global__ void hello_world() { printf("Hello world from the GPU\n"); }

__global__ void print_build_in_vars() {
  printf("%u gridDim.x=%d, blockDim.x=%d, blockIdx.x=%d, threadIdx.x=%d\n", device::get_grid_idx(), gridDim.x,
         blockDim.x, blockIdx.x, threadIdx.x);
}

__global__ void print_3d_grid_vars() {
  printf(
      "%u gridDim.[x=%d, y=%d, z=%d], "
      "blockDim.[x=%d, y=%d, z=%d], "
      "blockIdx.[x=%d, y=%d, z=%d], "
      "threadIdx.[x=%d, y=%d, z=%d]\n",
      device::get_thread_idx(),         //
      gridDim.x, gridDim.y, gridDim.z,     //
      blockDim.x, blockDim.y, blockDim.z,  //
      blockIdx.x, blockIdx.y, blockIdx.z,  //
      threadIdx.x, threadIdx.y, threadIdx.z);
}

__host__ void get_set_device() {
  int count = 0;
  auto result = cudaGetDeviceCount(&count);
  if (result == cudaSuccess) {
    printf("get device count success. [count=%d]\n", count);

    result = cudaSetDevice(count - 1);
    if (result == cudaSuccess) {
      printf("set cuda device success\n");
    }
  }
}