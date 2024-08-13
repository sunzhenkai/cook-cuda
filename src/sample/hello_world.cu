#include "hello_world.cuh"
#include "stdio.h"

__global__ void hello_world() { printf("Hello world from the GPU\n"); }

__global__ void print_build_in_vars() {
  printf("gridDim.x=%d, blockDim.x=%d, blockIdx.x=%d, threadIdx.x=%d\n", gridDim.x, blockDim.x, blockIdx.x,
         threadIdx.x);
}

__global__ void print_3d_grid_vars() {
  printf(
      "gridDim.[x=%d, y=%d, z=%d], "
      "blockDim.[x=%d, y=%d, z=%d], "
      "blockIdx.[x=%d, y=%d, z=%d], "
      "threadIdx.[x=%d, y=%d, z=%d]\n",
      gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z,
      threadIdx.x, threadIdx.y, threadIdx.z);
}