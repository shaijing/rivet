#include <stddef.h>
#include <stdint.h>

#include "cuda_utils.cuh"

template <typename T>
__device__ void copy_contiguous(const T *src, T *dst, size_t numel) {
    for (uint64_t index = rivet_global_index(); index < numel;
         index += rivet_global_stride()) {
        dst[index] = src[index];
    }
}

extern "C" __global__ void copy_f32(const float *src, float *dst, size_t numel) {
    copy_contiguous(src, dst, numel);
}

extern "C" __global__ void copy_u8(const uint8_t *src, uint8_t *dst, size_t numel) {
    copy_contiguous(src, dst, numel);
}

// Copy a logical tensor view into a contiguous output. `dims`, `src_strides`,
// and `dst_strides` are row-major element strides with `num_dims` entries.
extern "C" __global__ void copy_strided_f32(
    const float *src,
    float *dst,
    size_t numel,
    size_t num_dims,
    const size_t *dims,
    const size_t *src_strides,
    const size_t *dst_strides) {
    for (uint64_t linear = rivet_global_index(); linear < numel;
         linear += rivet_global_stride()) {
        size_t remaining = linear;
        size_t src_index = 0;
        size_t dst_index = 0;
        for (size_t dimension = num_dims; dimension-- > 0;) {
            const size_t coordinate = remaining % dims[dimension];
            remaining /= dims[dimension];
            src_index += coordinate * src_strides[dimension];
            dst_index += coordinate * dst_strides[dimension];
        }
        dst[dst_index] = src[src_index];
    }
}
