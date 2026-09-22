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

template <typename T>
__device__ void copy_layout(
    const T *src,
    T *dst,
    size_t numel,
    size_t num_dims,
    const size_t *info) {
    for (uint64_t linear = rivet_global_index(); linear < numel;
         linear += rivet_global_stride()) {
        dst[linear] = src[rivet_strided_index(linear, num_dims, info)];
    }
}

#define COPY_CONTIGUOUS(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *src, TYPENAME *dst, size_t numel) { \
    copy_contiguous(src, dst, numel); \
}

#define COPY_LAYOUT(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *info, \
    const TYPENAME *src, TYPENAME *dst) { \
    copy_layout(src, dst, numel, num_dims, info); \
}

COPY_CONTIGUOUS(uint8_t, copy_u8)
COPY_CONTIGUOUS(uint32_t, copy_u32)
COPY_CONTIGUOUS(int16_t, copy_i16)
COPY_CONTIGUOUS(int32_t, copy_i32)
COPY_CONTIGUOUS(int64_t, copy_i64)
COPY_CONTIGUOUS(float, copy_f32)
COPY_CONTIGUOUS(double, copy_f64)

COPY_LAYOUT(uint8_t, copy_layout_u8)
COPY_LAYOUT(uint32_t, copy_layout_u32)
COPY_LAYOUT(int16_t, copy_layout_i16)
COPY_LAYOUT(int32_t, copy_layout_i32)
COPY_LAYOUT(int64_t, copy_layout_i64)
COPY_LAYOUT(float, copy_layout_f32)
COPY_LAYOUT(double, copy_layout_f64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
COPY_CONTIGUOUS(__half, copy_f16)
COPY_LAYOUT(__half, copy_layout_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
COPY_CONTIGUOUS(__nv_bfloat16, copy_bf16)
COPY_LAYOUT(__nv_bfloat16, copy_layout_bf16)
#endif
