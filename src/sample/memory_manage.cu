#include "memory_manage.cuh"
#include "utils/utils.cuh"

#define ELEMENT_COUNT 100000

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
  //  common::Display(p_dest, ELEMENT_COUNT);

  // 4. 释放内存
  free(p_dest);
  free(p_a);
  free(p_b);
}
}  // namespace host

namespace device {
__device__ void matrix_sum(float *dest, float *a, float *b, size_t len) {}

__host__ void matrix_sum_entry() {
  // 1. 分配主机内存
  size_t byte_size = ELEMENT_COUNT * sizeof(float);
  float *p_h_dest, *p_h_a, *p_h_b;
  p_h_dest = (float *)malloc(byte_size);
  p_h_a = (float *)malloc(byte_size);
  p_h_b = (float *)malloc(byte_size);

  // 2. 分配设备内存
  float *p_d_dest, *p_d_a, *p_d_b;
  cudaMalloc((float **)&p_h_dest, byte_size);
  cudaMalloc((float **)&p_h_a, byte_size);
  cudaMalloc((float **)&p_h_b, byte_size);

  // 3. 初始化主机数据
  host::random_data(p_h_dest, byte_size);
  host::random_data(p_h_a, byte_size);
  host::random_data(p_h_b, byte_size);

  // 4. 从主机拷贝数据到设备
  cudaMemcpy(p_h_a, p_d_a, byte_size, cudaMemcpyHostToDevice);
}
}  // namespace device