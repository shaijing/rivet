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

#define UNARY_SCALAR_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *src, TYPENAME *dst, TYPENAME scalar, size_t numel) { \
    for (uint64_t index = rivet_global_index(); index < numel; \
         index += rivet_global_stride()) { \
        const TYPENAME x = src[index]; \
        dst[index] = EXPRESSION; \
    } \
}

#define UNARY_SCALAR_LAYOUT_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *info, \
    TYPENAME scalar, const TYPENAME *src, TYPENAME *dst) { \
    for (uint64_t linear = rivet_global_index(); linear < numel; \
         linear += rivet_global_stride()) { \
        const TYPENAME x = src[rivet_strided_index(linear, num_dims, info)]; \
        dst[linear] = EXPRESSION; \
    } \
}

template <typename T>
__device__ __forceinline__ T sign_value(T value) {
    return static_cast<T>(value > static_cast<T>(0)) -
           static_cast<T>(value < static_cast<T>(0));
}

#define BASIC_UNARY_SET(TYPENAME, SUFFIX) \
UNARY_OP(TYPENAME, recip_##SUFFIX, recipg(x)) \
UNARY_OP(TYPENAME, exp_##SUFFIX, expg(x)) \
UNARY_OP(TYPENAME, log_##SUFFIX, logg(x)) \
UNARY_OP(TYPENAME, sin_##SUFFIX, sing(x)) \
UNARY_OP(TYPENAME, cos_##SUFFIX, cosg(x)) \
UNARY_OP(TYPENAME, tanh_##SUFFIX, tanhg(x)) \
UNARY_OP(TYPENAME, erf_##SUFFIX, erfg(x)) \
UNARY_OP(TYPENAME, ceil_##SUFFIX, ceilg(x)) \
UNARY_OP(TYPENAME, floor_##SUFFIX, floorg(x)) \
UNARY_OP(TYPENAME, round_##SUFFIX, roundg(x)) \
UNARY_OP(TYPENAME, normcdf_##SUFFIX, normcdfg(x)) \
UNARY_OP(TYPENAME, square_##SUFFIX, x * x) \
UNARY_OP(TYPENAME, sqrt_##SUFFIX, sqrtg(x)) \
UNARY_OP(TYPENAME, sign_##SUFFIX, sign_value(x)) \
UNARY_LAYOUT_OP(TYPENAME, recip_layout_##SUFFIX, recipg(x)) \
UNARY_LAYOUT_OP(TYPENAME, exp_layout_##SUFFIX, expg(x)) \
UNARY_LAYOUT_OP(TYPENAME, log_layout_##SUFFIX, logg(x)) \
UNARY_LAYOUT_OP(TYPENAME, sin_layout_##SUFFIX, sing(x)) \
UNARY_LAYOUT_OP(TYPENAME, cos_layout_##SUFFIX, cosg(x)) \
UNARY_LAYOUT_OP(TYPENAME, tanh_layout_##SUFFIX, tanhg(x)) \
UNARY_LAYOUT_OP(TYPENAME, erf_layout_##SUFFIX, erfg(x)) \
UNARY_LAYOUT_OP(TYPENAME, ceil_layout_##SUFFIX, ceilg(x)) \
UNARY_LAYOUT_OP(TYPENAME, floor_layout_##SUFFIX, floorg(x)) \
UNARY_LAYOUT_OP(TYPENAME, round_layout_##SUFFIX, roundg(x)) \
UNARY_LAYOUT_OP(TYPENAME, normcdf_layout_##SUFFIX, normcdfg(x)) \
UNARY_LAYOUT_OP(TYPENAME, square_layout_##SUFFIX, x * x) \
UNARY_LAYOUT_OP(TYPENAME, sqrt_layout_##SUFFIX, sqrtg(x)) \
UNARY_LAYOUT_OP(TYPENAME, sign_layout_##SUFFIX, sign_value(x)) \
UNARY_SCALAR_OP(TYPENAME, pow_scalar_##SUFFIX, powg(x, scalar)) \
UNARY_SCALAR_LAYOUT_OP(TYPENAME, pow_scalar_layout_##SUFFIX, powg(x, scalar))

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

BASIC_UNARY_SET(float, f32)
BASIC_UNARY_SET(double, f64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
UNARY_PAIR(__half, f16, -x, (x < __half(0.0f) ? -x : x))
BASIC_UNARY_SET(__half, f16)
#endif

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
UNARY_PAIR(__nv_bfloat16, bf16, -x, (x < __nv_bfloat16(0.0f) ? -x : x))
BASIC_UNARY_SET(__nv_bfloat16, bf16)
#endif
