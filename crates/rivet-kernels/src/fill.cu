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

#define FILL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    TYPENAME *dst, TYPENAME value, size_t numel) { \
    fill_contiguous(dst, value, numel); \
}

template <typename T>
__device__ void copy2d(
    const T *src,
    T *dst,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t src_stride,
    uint32_t dst_stride) {
    const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= dim0 * dim1) {
        return;
    }
    const uint32_t row = linear / dim1;
    const uint32_t column = linear - dim1 * row;
    dst[row * dst_stride + column] = src[row * src_stride + column];
}

#define COPY2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *src, TYPENAME *dst, uint32_t dim0, uint32_t dim1, \
    uint32_t src_stride, uint32_t dst_stride) { \
    copy2d(src, dst, dim0, dim1, src_stride, dst_stride); \
}

#define CONST_SET_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *info, \
    TYPENAME value, TYPENAME *dst) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (uint64_t index = rivet_global_index(); index < numel; \
             index += rivet_global_stride()) { \
            dst[index] = value; \
        } \
    } else { \
        for (uint64_t index = rivet_global_index(); index < numel; \
             index += rivet_global_stride()) { \
            dst[get_strided_index(index, num_dims, dims, strides)] = value; \
        } \
    } \
}

FILL_OP(uint8_t, fill_u8)
FILL_OP(uint32_t, fill_u32)
FILL_OP(int16_t, fill_i16)
FILL_OP(int32_t, fill_i32)
FILL_OP(int64_t, fill_i64)
FILL_OP(float, fill_f32)
FILL_OP(double, fill_f64)

COPY2D_OP(float, copy2d_f32)
COPY2D_OP(double, copy2d_f64)
COPY2D_OP(uint8_t, copy2d_u8)
COPY2D_OP(uint32_t, copy2d_u32)
COPY2D_OP(int16_t, copy2d_i16)
COPY2D_OP(int32_t, copy2d_i32)
COPY2D_OP(int64_t, copy2d_i64)

CONST_SET_OP(float, const_set_f32)
CONST_SET_OP(double, const_set_f64)
CONST_SET_OP(uint8_t, const_set_u8)
CONST_SET_OP(uint32_t, const_set_u32)
CONST_SET_OP(int16_t, const_set_i16)
CONST_SET_OP(int32_t, const_set_i32)
CONST_SET_OP(int64_t, const_set_i64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
FILL_OP(__half, fill_f16)
COPY2D_OP(__half, copy2d_f16)
CONST_SET_OP(__half, const_set_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
FILL_OP(__nv_bfloat16, fill_bf16)
COPY2D_OP(__nv_bfloat16, copy2d_bf16)
CONST_SET_OP(__nv_bfloat16, const_set_bf16)

#include "cuda_fp8.h"
FILL_OP(__nv_fp8_e4m3, fill_f8_e4m3)
COPY2D_OP(__nv_fp8_e4m3, copy2d_f8_e4m3)
CONST_SET_OP(__nv_fp8_e4m3, const_set_f8_e4m3)
#endif
