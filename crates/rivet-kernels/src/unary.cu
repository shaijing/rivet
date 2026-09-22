#include <stddef.h>

#include "cuda_utils.cuh"

extern "C" __global__ void neg_f32(const float *src, float *dst, size_t numel) {
    for (uint64_t index = rivet_global_index(); index < numel;
         index += rivet_global_stride()) {
        dst[index] = -src[index];
    }
}

extern "C" __global__ void identity_f32(const float *src, float *dst, size_t numel) {
    for (uint64_t index = rivet_global_index(); index < numel;
         index += rivet_global_stride()) {
        dst[index] = src[index];
    }
}
