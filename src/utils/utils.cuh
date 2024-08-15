#pragma once

#include "cuda_runtime.h"
#include "iostream"
#include "vector"

namespace common {
__host__ __device__ void Display(const std::vector<float> &v);
__host__ __device__ void Display(const float *f, size_t len);
}  // namespace common

namespace host {
__host__ std::vector<std::string> GetDevices();
__host__ void PrintDeviceInfo();
}  // namespace host

namespace device {
__device__ uint get_grid_idx();
__device__ uint get_block_idx();
__device__ uint get_thread_idx();
}
