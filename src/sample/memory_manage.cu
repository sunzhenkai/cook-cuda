#include <assert.h>

#include "memory_manage.cuh"
#include "utils/utils.cuh"

#define ELEMENT_COUNT 100

namespace host {
__host__ void matrix_sum(float *dest, float *a, float *b, size_t len) {
  for (size_t i = 0ul; i < len; ++i) {
    dest[i] = a[i] + b[i];
  }
}

__host__ void random_data(float *a, size_t len) {
  for (size_t i = 0ul; i < len; ++i) {
    a[i] = (float)(rand() & 0xff) / 3.0f;
  }
}

__host__ void matrix_sum_entry() {
  // 1. 分配内存
  size_t byte_size = ELEMENT_COUNT * sizeof(float);
  float *p_dest, *p_a, *p_b;
  p_dest = (float *)malloc(byte_size);
  p_a = (float *)malloc(byte_size);
  p_b = (float *)malloc(byte_size);

  // 2. 初始化数值
  srand(100);
  random_data(p_dest, ELEMENT_COUNT);
  random_data(p_a, ELEMENT_COUNT);
  random_data(p_b, ELEMENT_COUNT);

  // 3. 计算
  matrix_sum(p_dest, p_a, p_b, ELEMENT_COUNT);
  //  common::display(p_dest, ELEMENT_COUNT);

  // 4. 释放内存
  free(p_dest);
  free(p_a);
  free(p_b);
}
}  // namespace host

namespace device {
__global__ void matrix_sum(float *dest, float *a, float *b, size_t len) {
  uint start, end;
  get_task_range(&start, &end, len);
  //  printf("%u, [%u, %u]\n", get_thread_idx(), start, end);

  for (uint i = start; i < end; ++i) {
    dest[i] = a[i] + b[i];
    //    printf("%f = %f + %f\n", dest[i], a[i], b[i]);
  }
}

__host__ void matrix_sum_entry() {
  assert(cudaSetDevice(0) == cudaSuccess);

  // 1. 分配主机内存
  size_t byte_size = ELEMENT_COUNT * sizeof(float);
  float *p_h_dest, *p_h_a, *p_h_b;
  p_h_dest = (float *)malloc(byte_size);
  p_h_a = (float *)malloc(byte_size);
  p_h_b = (float *)malloc(byte_size);

  // 2. 分配设备内存
  float *p_d_dest, *p_d_a, *p_d_b;
  cudaMalloc((float **)&p_d_dest, byte_size);
  cudaMalloc((float **)&p_d_a, byte_size);
  cudaMalloc((float **)&p_d_b, byte_size);

  // 3. 初始化主机数据
  memset(p_h_dest, 0, byte_size);
  host::random_data(p_h_a, ELEMENT_COUNT);
  host::random_data(p_h_b, ELEMENT_COUNT);
  //  common::display(p_h_a, 10);
  //  common::display(p_h_b, 10);

  // 4. 从主机拷贝数据到设备
  cudaMemset(p_d_dest, 0, byte_size);
  cudaMemcpy(p_d_a, p_h_a, byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(p_d_b, p_h_b, byte_size, cudaMemcpyHostToDevice);

  // 5. 调用核函数计算
  cudaEvent_t start, end;
  float elapsed_time_ms;
  common::perf_start(&start, &end);
  matrix_sum<<<2, 5>>>(p_d_dest, p_d_a, p_d_b, ELEMENT_COUNT);
  common::perf_end(&elapsed_time_ms, &start, &end);
  printf("elapsed time: %f ms\n", elapsed_time_ms);

  // 6. 从设备拷贝结果到主机
  cudaMemcpy(p_h_dest, p_d_dest, byte_size, cudaMemcpyDeviceToHost);
  common::display(p_h_dest, 10);

  // 7. 释放内存
  free(p_h_dest);
  free(p_h_a);
  free(p_h_b);
  cudaFree(p_d_dest);
  cudaFree(p_d_a);
  cudaFree(p_d_b);

  cudaDeviceReset();
}

//__host__ void matrix_sum_prof() { // 问题: 需要 setDevice 及 unset Device
//  cudaEvent_t start, end;
//  common::error_check(cudaEventCreate(&start), __FILE__, __LINE__);
//  common::error_check(cudaEventCreate(&end), __FILE__, __LINE__);
//  common::error_check(cudaEventRecord(start), __FILE__, __LINE__);
//  cudaEventQuery(start);
//
//  matrix_sum_entry();
//
//  common::error_check(cudaEventRecord(end), __FILE__, __LINE__);
//  common::error_check(cudaEventSynchronize(end), __FILE__, __LINE__);
//  float elapsed_time_ms;
//  ERROR_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, end));
//
//  printf("elapsed time: %f ms", elapsed_time_ms);
//  ERROR_CHECK(cudaEventDestroy(start));
//  ERROR_CHECK(cudaEventDestroy(end));
//}
}  // namespace device