#pragma once

#include "cuda_runtime.h"
#include "iostream"
#include "vector"

namespace common {
__host__ __device__ void display(const std::vector<float> &v);
__host__ __device__ void display(const float *f, size_t len);
__host__ __device__ cudaError_t error_check(cudaError_t err, const char *fn, int line);
#define ERROR_CHECK(exp) common::error_check(exp, __FILE__, __LINE__)
__host__ __device__ void perf_start(cudaEvent_t *start, cudaEvent_t *end);
__host__ __device__ void perf_end(float *elapsed_time_ms, cudaEvent_t *start, cudaEvent_t *end);
}  // namespace common

namespace host {
__host__ std::vector<std::string> GetDevices();
__host__ void PrintDeviceInfo();
}  // namespace host

namespace device {
__device__ uint get_grid_idx();
__device__ uint get_block_idx();
__device__ uint get_thread_idx();
__device__ uint get_total_thread();
__device__ void get_task_range(uint *start, uint *end, uint len);
}  // namespace device
