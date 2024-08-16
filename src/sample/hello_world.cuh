/**
 * tensor examples
 * __global__ : 核函数 (kernel function)
 */

__global__ void hello_world();
__global__ void print_build_in_vars();
__global__ void print_3d_grid_vars();
__host__ void get_set_device();
__host__ __device__ cudaError_t error_check(cudaError_t err, const char *fn, int line);
__host__ void error_check_entry();
__global__ void kernel_error_entry();