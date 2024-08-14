#include "utils.cuh"

namespace common {
__host__ __device__ void Display(const std::vector<float> &v) {
  for (auto t : v) {
    printf("%f ", t);
  }
  printf("\n");
}

__host__ __device__ void Display(const float *f, size_t len) {
  for (size_t i = 0ul; i < len; ++i) printf("%f", *(f + i));
  printf("\n");
}
}  // namespace common

namespace host {
__host__ std::vector<std::string> GetDevices() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::vector<std::string> result;
  result.reserve(deviceCount);

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp dp{};
    cudaGetDeviceProperties(&dp, i);
    result.emplace_back(dp.name);
  }
  return std::move(result);
}

__host__ void PrintDeviceInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "GPU device count: " << deviceCount << std::endl;

  for (int i = 0; i < deviceCount; ++i) {
    // sm: 流式多处理器, Streaming Multiprocessor
    cudaDeviceProp dp{};
    cudaGetDeviceProperties(&dp, i);
    std::cout << "device.0  " << std::endl;
    std::cout << "  sm count: \t\t\t\t" << dp.multiProcessorCount << std::endl;
    std::cout << "  shared memory per block: \t\t" << dp.sharedMemPerBlock / 1024 << "KB" << std::endl;
    std::cout << "  max threads per block:\t\t" << dp.maxThreadsPerBlock << std::endl;
    std::cout << "  max threads per multi processor:\t" << dp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  max threads per sm:\t\t\t" << dp.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << "  max blocks per multi processor:\t" << dp.maxBlocksPerMultiProcessor << std::endl;
  }
}
}  // namespace host
