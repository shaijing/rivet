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

FILL_OP(uint8_t, fill_u8)
FILL_OP(uint32_t, fill_u32)
FILL_OP(int16_t, fill_i16)
FILL_OP(int32_t, fill_i32)
FILL_OP(int64_t, fill_i64)
FILL_OP(float, fill_f32)
FILL_OP(double, fill_f64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
FILL_OP(__half, fill_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
FILL_OP(__nv_bfloat16, fill_bf16)
#endif
