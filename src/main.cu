#include "sample/hello_world.cuh"
#include "sample/memory_manage.cuh"
#include "utils/utils.cuh"

void f0() {
  host::PrintDeviceInfo();
  hello_world<<<1, 1>>>();
  hello_world<<<2, 4>>>();
  print_build_in_vars<<<2, 4>>>();
  print_3d_grid_vars<<<dim3{2, 2, 1}, dim3{3, 2, 1}>>>();

  cudaDeviceSynchronize();  // 同步等待 GPU 执行核函数

  get_set_device();
}

void f1() {
  host::matrix_sum_entry();
  device::matrix_sum_entry();
}

void f2() {
  error_check_entry();
  kernel_error_entry<<<1, 1>>>();
  error_check(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

void f3() { device::matrix_sum_entry(); }

int main() {
  //    f0();
  //  f1();
  f3();
  return 0;
}
