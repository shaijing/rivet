#include <stddef.h>
#include <stdint.h>

#include "cuda_utils.cuh"

template <typename T>
__device__ void fill_contiguous(T *dst, T value, size_t numel) {
    for (uint64_t index = rivet_global_index(); index < numel;
         index += rivet_global_stride()) {
        dst[index] = value;
    }
}

extern "C" __global__ void fill_f32(float *dst, float value, size_t numel) {
    fill_contiguous(dst, value, numel);
}

extern "C" __global__ void fill_u8(uint8_t *dst, uint8_t value, size_t numel) {
    fill_contiguous(dst, value, numel);
}
