#pragma once

namespace host {
__host__ void matrix_sum(float *dest, float *a, float *b, size_t len);
__host__ void matrix_sum_entry();
}  // namespace cpu

namespace device {}