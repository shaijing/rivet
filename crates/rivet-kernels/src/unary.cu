#include <stddef.h>
#include <stdint.h>

#include "cuda_utils.cuh"

#define UNARY_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *src, TYPENAME *dst, size_t numel) { \
    for (uint64_t index = rivet_global_index(); index < numel; \
         index += rivet_global_stride()) { \
        const TYPENAME x = src[index]; \
        dst[index] = EXPRESSION; \
    } \
}

#define UNARY_LAYOUT_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *info, \
    const TYPENAME *src, TYPENAME *dst) { \
    for (uint64_t linear = rivet_global_index(); linear < numel; \
         linear += rivet_global_stride()) { \
        const TYPENAME x = src[rivet_strided_index(linear, num_dims, info)]; \
        dst[linear] = EXPRESSION; \
    } \
}

#define UNARY_PAIR(TYPENAME, SUFFIX, NEG, ABS) \
UNARY_OP(TYPENAME, neg_##SUFFIX, NEG) \
UNARY_OP(TYPENAME, abs_##SUFFIX, ABS) \
UNARY_LAYOUT_OP(TYPENAME, neg_layout_##SUFFIX, NEG) \
UNARY_LAYOUT_OP(TYPENAME, abs_layout_##SUFFIX, ABS)

UNARY_PAIR(uint8_t, u8, (uint8_t)(0 - x), x)
UNARY_PAIR(uint32_t, u32, (uint32_t)(0 - x), x)
UNARY_PAIR(int16_t, i16, (int16_t)(0 - x), (x < 0 ? -x : x))
UNARY_PAIR(int32_t, i32, (int32_t)(0 - x), (x < 0 ? -x : x))
UNARY_PAIR(int64_t, i64, (int64_t)(0 - x), (x < 0 ? -x : x))
UNARY_PAIR(float, f32, -x, (x < 0.0f ? -x : x))
UNARY_PAIR(double, f64, -x, (x < 0.0 ? -x : x))

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
UNARY_PAIR(__half, f16, -x, (x < __half(0.0f) ? -x : x))
#endif

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
UNARY_PAIR(__nv_bfloat16, bf16, -x, (x < __nv_bfloat16(0.0f) ? -x : x))
#endif
