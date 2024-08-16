#include <cstdio>

#include "hello_world.cuh"
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
      device::get_thread_idx(),            //
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

__host__ __device__ cudaError_t error_check(cudaError_t err, const char *fn, int line) {
  if (err != cudaSuccess) {
    printf("CUDA error:\n\tcode=%d, name=%s, description=%s, \n\tfile=%s, line=%d\n", err, cudaGetErrorName(err),
           cudaGetErrorString(err), fn, line);
  } else {
    printf("%s, ok\n", __func__);
  }
  return err;
}

__host__ void error_check_entry() {
  int device_id_in_use;
  error_check(cudaGetDevice(&device_id_in_use), __FILE__, __LINE__);
  error_check(cudaSetDevice(999), __FILE__, __LINE__);
  //  char *p_c;
  //  error_check(cudaMalloc(&p_c, 100), __FILE__, __LINE__);

  cudaDeviceSynchronize();
} /** output
error_check, ok
CUDA error:
        code=101, name=cudaErrorInvalidDevice, description=invalid device ordinal,
        file=/data/code/cook-cuda/src/sample/hello_world.cu, line=51
*/

__global__ void kernel_error_entry() {
  dim3 block(2048);
  print_build_in_vars<<<2, block>>>();  // block size 最大 1024
  error_check(cudaGetLastError(), __FILE__, __LINE__);
} /** output
CUDA error:
        code=9, name=cudaErrorInvalidConfiguration, description=invalid configuration argument,
        file=/data/code/cook-cuda/src/sample/hello_world.cu, line=67
*/