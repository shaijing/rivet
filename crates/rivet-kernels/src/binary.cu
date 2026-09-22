#include <stddef.h>
#include <stdint.h>

#include "cuda_utils.cuh"

#define BINARY_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *lhs, const TYPENAME *rhs, TYPENAME *dst, size_t numel) { \
    for (uint64_t index = rivet_global_index(); index < numel; \
         index += rivet_global_stride()) { \
        const TYPENAME x = lhs[index]; \
        const TYPENAME y = rhs[index]; \
        dst[index] = EXPRESSION; \
    } \
}

#define BINARY_LAYOUT_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *lhs_info, \
    const size_t *rhs_info, const TYPENAME *lhs, const TYPENAME *rhs, \
    TYPENAME *dst) { \
    for (uint64_t linear = rivet_global_index(); linear < numel; \
         linear += rivet_global_stride()) { \
        const TYPENAME x = lhs[rivet_strided_index(linear, num_dims, lhs_info)]; \
        const TYPENAME y = rhs[rivet_strided_index(linear, num_dims, rhs_info)]; \
        dst[linear] = EXPRESSION; \
    } \
}

#define BINARY_SCALAR_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *src, TYPENAME *dst, TYPENAME scalar, size_t numel) { \
    for (uint64_t index = rivet_global_index(); index < numel; \
         index += rivet_global_stride()) { \
        const TYPENAME x = src[index]; \
        const TYPENAME y = scalar; \
        dst[index] = EXPRESSION; \
    } \
}

#define BINARY_SCALAR_LAYOUT_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *info, TYPENAME scalar, \
    const TYPENAME *src, TYPENAME *dst) { \
    for (uint64_t linear = rivet_global_index(); linear < numel; \
         linear += rivet_global_stride()) { \
        const TYPENAME x = src[rivet_strided_index(linear, num_dims, info)]; \
        const TYPENAME y = scalar; \
        dst[linear] = EXPRESSION; \
    } \
}

#define COMPARE_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *lhs, const TYPENAME *rhs, uint8_t *dst, size_t numel) { \
    for (uint64_t index = rivet_global_index(); index < numel; \
         index += rivet_global_stride()) { \
        const TYPENAME x = lhs[index]; \
        const TYPENAME y = rhs[index]; \
        dst[index] = (uint8_t)(EXPRESSION); \
    } \
}

#define COMPARE_LAYOUT_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *lhs_info, \
    const size_t *rhs_info, const TYPENAME *lhs, const TYPENAME *rhs, \
    uint8_t *dst) { \
    for (uint64_t linear = rivet_global_index(); linear < numel; \
         linear += rivet_global_stride()) { \
        const TYPENAME x = lhs[rivet_strided_index(linear, num_dims, lhs_info)]; \
        const TYPENAME y = rhs[rivet_strided_index(linear, num_dims, rhs_info)]; \
        dst[linear] = (uint8_t)(EXPRESSION); \
    } \
}

#define COMPARE_SCALAR_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *src, uint8_t *dst, TYPENAME scalar, size_t numel) { \
    for (uint64_t index = rivet_global_index(); index < numel; \
         index += rivet_global_stride()) { \
        const TYPENAME x = src[index]; \
        const TYPENAME y = scalar; \
        dst[index] = (uint8_t)(EXPRESSION); \
    } \
}

#define COMPARE_SCALAR_LAYOUT_OP(TYPENAME, FN_NAME, EXPRESSION) \
extern "C" __global__ void FN_NAME( \
    size_t numel, size_t num_dims, const size_t *info, TYPENAME scalar, \
    const TYPENAME *src, uint8_t *dst) { \
    for (uint64_t linear = rivet_global_index(); linear < numel; \
         linear += rivet_global_stride()) { \
        const TYPENAME x = src[rivet_strided_index(linear, num_dims, info)]; \
        const TYPENAME y = scalar; \
        dst[linear] = (uint8_t)(EXPRESSION); \
    } \
}

#define BINARY_SET(TYPENAME, SUFFIX) \
BINARY_OP(TYPENAME, add_##SUFFIX, x + y) \
BINARY_OP(TYPENAME, sub_##SUFFIX, x - y) \
BINARY_OP(TYPENAME, mul_##SUFFIX, x * y) \
BINARY_OP(TYPENAME, div_##SUFFIX, x / y) \
BINARY_OP(TYPENAME, minimum_##SUFFIX, (x < y ? x : y)) \
BINARY_OP(TYPENAME, maximum_##SUFFIX, (x > y ? x : y)) \
BINARY_LAYOUT_OP(TYPENAME, add_layout_##SUFFIX, x + y) \
BINARY_LAYOUT_OP(TYPENAME, sub_layout_##SUFFIX, x - y) \
BINARY_LAYOUT_OP(TYPENAME, mul_layout_##SUFFIX, x * y) \
BINARY_LAYOUT_OP(TYPENAME, div_layout_##SUFFIX, x / y) \
BINARY_LAYOUT_OP(TYPENAME, minimum_layout_##SUFFIX, (x < y ? x : y)) \
BINARY_LAYOUT_OP(TYPENAME, maximum_layout_##SUFFIX, (x > y ? x : y)) \
BINARY_SCALAR_OP(TYPENAME, add_scalar_##SUFFIX, x + y) \
BINARY_SCALAR_OP(TYPENAME, sub_scalar_##SUFFIX, x - y) \
BINARY_SCALAR_OP(TYPENAME, mul_scalar_##SUFFIX, x * y) \
BINARY_SCALAR_OP(TYPENAME, div_scalar_##SUFFIX, x / y) \
BINARY_SCALAR_OP(TYPENAME, minimum_scalar_##SUFFIX, (x < y ? x : y)) \
BINARY_SCALAR_OP(TYPENAME, maximum_scalar_##SUFFIX, (x > y ? x : y)) \
BINARY_SCALAR_LAYOUT_OP(TYPENAME, add_scalar_layout_##SUFFIX, x + y) \
BINARY_SCALAR_LAYOUT_OP(TYPENAME, sub_scalar_layout_##SUFFIX, x - y) \
BINARY_SCALAR_LAYOUT_OP(TYPENAME, mul_scalar_layout_##SUFFIX, x * y) \
BINARY_SCALAR_LAYOUT_OP(TYPENAME, div_scalar_layout_##SUFFIX, x / y) \
BINARY_SCALAR_LAYOUT_OP(TYPENAME, minimum_scalar_layout_##SUFFIX, (x < y ? x : y)) \
BINARY_SCALAR_LAYOUT_OP(TYPENAME, maximum_scalar_layout_##SUFFIX, (x > y ? x : y))

#define COMPARE_SET(TYPENAME, SUFFIX) \
COMPARE_OP(TYPENAME, eq_##SUFFIX, x == y) \
COMPARE_OP(TYPENAME, ne_##SUFFIX, x != y) \
COMPARE_OP(TYPENAME, lt_##SUFFIX, x < y) \
COMPARE_OP(TYPENAME, le_##SUFFIX, x <= y) \
COMPARE_OP(TYPENAME, gt_##SUFFIX, x > y) \
COMPARE_OP(TYPENAME, ge_##SUFFIX, x >= y) \
COMPARE_LAYOUT_OP(TYPENAME, eq_layout_##SUFFIX, x == y) \
COMPARE_LAYOUT_OP(TYPENAME, ne_layout_##SUFFIX, x != y) \
COMPARE_LAYOUT_OP(TYPENAME, lt_layout_##SUFFIX, x < y) \
COMPARE_LAYOUT_OP(TYPENAME, le_layout_##SUFFIX, x <= y) \
COMPARE_LAYOUT_OP(TYPENAME, gt_layout_##SUFFIX, x > y) \
COMPARE_LAYOUT_OP(TYPENAME, ge_layout_##SUFFIX, x >= y) \
COMPARE_SCALAR_OP(TYPENAME, eq_scalar_##SUFFIX, x == y) \
COMPARE_SCALAR_OP(TYPENAME, ne_scalar_##SUFFIX, x != y) \
COMPARE_SCALAR_OP(TYPENAME, lt_scalar_##SUFFIX, x < y) \
COMPARE_SCALAR_OP(TYPENAME, le_scalar_##SUFFIX, x <= y) \
COMPARE_SCALAR_OP(TYPENAME, gt_scalar_##SUFFIX, x > y) \
COMPARE_SCALAR_OP(TYPENAME, ge_scalar_##SUFFIX, x >= y) \
COMPARE_SCALAR_LAYOUT_OP(TYPENAME, eq_scalar_layout_##SUFFIX, x == y) \
COMPARE_SCALAR_LAYOUT_OP(TYPENAME, ne_scalar_layout_##SUFFIX, x != y) \
COMPARE_SCALAR_LAYOUT_OP(TYPENAME, lt_scalar_layout_##SUFFIX, x < y) \
COMPARE_SCALAR_LAYOUT_OP(TYPENAME, le_scalar_layout_##SUFFIX, x <= y) \
COMPARE_SCALAR_LAYOUT_OP(TYPENAME, gt_scalar_layout_##SUFFIX, x > y) \
COMPARE_SCALAR_LAYOUT_OP(TYPENAME, ge_scalar_layout_##SUFFIX, x >= y)

BINARY_SET(uint8_t, u8)
COMPARE_SET(uint8_t, u8)
BINARY_SET(uint32_t, u32)
COMPARE_SET(uint32_t, u32)
BINARY_SET(int16_t, i16)
COMPARE_SET(int16_t, i16)
BINARY_SET(int32_t, i32)
COMPARE_SET(int32_t, i32)
BINARY_SET(int64_t, i64)
COMPARE_SET(int64_t, i64)
BINARY_SET(float, f32)
COMPARE_SET(float, f32)
BINARY_SET(double, f64)
COMPARE_SET(double, f64)

#if __CUDA_ARCH__ >= 530
#include "cuda_fp16.h"
BINARY_SET(__half, f16)
COMPARE_SET(__half, f16)
#endif

#if __CUDA_ARCH__ >= 800
#include "cuda_bf16.h"
BINARY_SET(__nv_bfloat16, bf16)
COMPARE_SET(__nv_bfloat16, bf16)
#endif
