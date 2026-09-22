#include <stddef.h>

#include "cuda_utils.cuh"

extern "C" __global__ void add_f32(
    const float *lhs,
    const float *rhs,
    float *dst,
    size_t numel) {
    for (uint64_t index = rivet_global_index(); index < numel;
         index += rivet_global_stride()) {
        dst[index] = lhs[index] + rhs[index];
    }
}

extern "C" __global__ void mul_f32(
    const float *lhs,
    const float *rhs,
    float *dst,
    size_t numel) {
    for (uint64_t index = rivet_global_index(); index < numel;
         index += rivet_global_stride()) {
        dst[index] = lhs[index] * rhs[index];
    }
}
