#pragma once

namespace host {
__host__ void matrix_sum(float *dest, float *a, float *b, size_t len);
__host__ void matrix_sum_entry();
}  // namespace host

namespace device {

__global__ void matrix_sum(float *dest, float *a, float *b, size_t len);
__host__ void matrix_sum_entry();
//__host__ void matrix_sum_prof();
}  // namespace device