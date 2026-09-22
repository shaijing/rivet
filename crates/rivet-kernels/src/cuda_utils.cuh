#pragma once

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "compatibility.cuh"

__device__ __forceinline__ uint64_t rivet_global_index() {
    return static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint64_t rivet_global_stride() {
    return static_cast<uint64_t>(blockDim.x) * gridDim.x;
}

__device__ __forceinline__ size_t rivet_strided_index(
    size_t linear,
    size_t num_dims,
    const size_t *info) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    size_t remaining = linear;
    size_t index = 0;
    for (size_t dimension = num_dims; dimension-- > 0;) {
        const size_t coordinate = remaining % dims[dimension];
        remaining /= dims[dimension];
        index += coordinate * strides[dimension];
    }
    return index;
}

// Candle-compatible layout helpers used by the migrated basic kernels. The
// info buffer stores `dims` followed by `strides`.
__device__ __forceinline__ bool is_contiguous(
    size_t num_dims,
    const size_t *dims,
    const size_t *strides) {
    size_t expected = 1;
    for (size_t dimension = num_dims; dimension-- > 0;) {
        if (dims[dimension] > 1 && strides[dimension] != expected) {
            return false;
        }
        expected *= dims[dimension];
    }
    return true;
}

__device__ __forceinline__ unsigned int get_strided_index(
    unsigned int linear,
    size_t num_dims,
    const size_t *dims,
    const size_t *strides) {
    unsigned int index = 0;
    for (size_t dimension = num_dims; dimension-- > 0;) {
        index += (linear % dims[dimension]) * strides[dimension];
        linear /= dims[dimension];
    }
    return index;
}

__device__ __forceinline__ unsigned int next_power_of_two(unsigned int value) {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

__device__ __forceinline__ uint8_t ming(uint8_t lhs, uint8_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ uint8_t maxg(uint8_t lhs, uint8_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ uint32_t ming(uint32_t lhs, uint32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ uint32_t maxg(uint32_t lhs, uint32_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int16_t ming(int16_t lhs, int16_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ int16_t maxg(int16_t lhs, int16_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int32_t ming(int32_t lhs, int32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ int32_t maxg(int32_t lhs, int32_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int64_t ming(int64_t lhs, int64_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ int64_t maxg(int64_t lhs, int64_t rhs) {
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ float ming(float lhs, float rhs) {
    return fminf(lhs, rhs);
}

__device__ __forceinline__ float maxg(float lhs, float rhs) {
    return fmaxf(lhs, rhs);
}

__device__ __forceinline__ double ming(double lhs, double rhs) {
    return fmin(lhs, rhs);
}

__device__ __forceinline__ double maxg(double lhs, double rhs) {
    return fmax(lhs, rhs);
}

__device__ __forceinline__ bool isnang(float value) { return isnan(value); }
__device__ __forceinline__ bool isnang(double value) { return isnan(value); }
__device__ __forceinline__ float recipg(float value) { return 1.0f / value; }
__device__ __forceinline__ double recipg(double value) { return 1.0 / value; }
__device__ __forceinline__ float cosg(float value) { return cosf(value); }
__device__ __forceinline__ double cosg(double value) { return cos(value); }
__device__ __forceinline__ float sing(float value) { return sinf(value); }
__device__ __forceinline__ double sing(double value) { return sin(value); }
__device__ __forceinline__ float sqrtg(float value) { return sqrtf(value); }
__device__ __forceinline__ double sqrtg(double value) { return sqrt(value); }
__device__ __forceinline__ float powg(float lhs, float rhs) { return powf(lhs, rhs); }
__device__ __forceinline__ double powg(double lhs, double rhs) { return pow(lhs, rhs); }
__device__ __forceinline__ float tanhg(float value) { return tanhf(value); }
__device__ __forceinline__ double tanhg(double value) { return tanh(value); }
__device__ __forceinline__ float erfg(float value) { return erff(value); }
__device__ __forceinline__ double erfg(double value) { return erf(value); }
__device__ __forceinline__ float ceilg(float value) { return ceilf(value); }
__device__ __forceinline__ double ceilg(double value) { return ceil(value); }
__device__ __forceinline__ float floorg(float value) { return floorf(value); }
__device__ __forceinline__ double floorg(double value) { return floor(value); }
__device__ __forceinline__ float roundg(float value) { return roundf(value); }
__device__ __forceinline__ double roundg(double value) { return round(value); }
__device__ __forceinline__ float normcdfg(float value) { return normcdff(value); }
__device__ __forceinline__ double normcdfg(double value) { return normcdf(value); }
__device__ __forceinline__ float logg(float value) { return logf(value); }
__device__ __forceinline__ double logg(double value) { return log(value); }
__device__ __forceinline__ float expg(float value) { return expf(value); }
__device__ __forceinline__ double expg(double value) { return exp(value); }
__device__ __forceinline__ float absg(float value) { return fabsf(value); }
__device__ __forceinline__ double absg(double value) { return fabs(value); }
__device__ __forceinline__ float copysigng(float lhs, float rhs) {
    return copysignf(lhs, rhs);
}
__device__ __forceinline__ double copysigng(double lhs, double rhs) {
    return copysign(lhs, rhs);
}

#if __CUDA_ARCH__ >= 530
__device__ __forceinline__ __half ming(__half lhs, __half rhs) {
    return __hmin_nan(lhs, rhs);
}

__device__ __forceinline__ __half maxg(__half lhs, __half rhs) {
    return __hmax_nan(lhs, rhs);
}

__device__ __forceinline__ __half powg(__half lhs, __half rhs) {
    return __float2half(powf(__half2float(lhs), __half2float(rhs)));
}
__device__ __forceinline__ bool isnang(__half value) { return __hisnan(value); }
__device__ __forceinline__ __half sqrtg(__half value) { return hsqrt(value); }
__device__ __forceinline__ __half cosg(__half value) { return hcos(value); }
__device__ __forceinline__ __half sing(__half value) { return hsin(value); }
__device__ __forceinline__ __half recipg(__half value) {
    return __half(1.0f) / value;
}
__device__ __forceinline__ __half tanhg(__half value) {
    return __float2half(tanhf(__half2float(value)));
}
__device__ __forceinline__ __half erfg(__half value) {
    return __float2half(erff(__half2float(value)));
}
__device__ __forceinline__ __half ceilg(__half value) {
    return __float2half(ceilf(__half2float(value)));
}
__device__ __forceinline__ __half floorg(__half value) {
    return __float2half(floorf(__half2float(value)));
}
__device__ __forceinline__ __half roundg(__half value) {
    return __float2half(roundf(__half2float(value)));
}
__device__ __forceinline__ __half normcdfg(__half value) {
    return __float2half(normcdff(__half2float(value)));
}
__device__ __forceinline__ __half logg(__half value) { return hlog(value); }
__device__ __forceinline__ __half expg(__half value) { return hexp(value); }
__device__ __forceinline__ __half absg(__half value) { return __habs(value); }
__device__ __forceinline__ __half copysigng(__half lhs, __half rhs) {
    return __float2half(copysignf(__half2float(lhs), __half2float(rhs)));
}
#endif

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ __nv_bfloat16 ming(
    __nv_bfloat16 lhs,
    __nv_bfloat16 rhs) {
    return __hmin_nan(lhs, rhs);
}

__device__ __forceinline__ __nv_bfloat16 maxg(
    __nv_bfloat16 lhs,
    __nv_bfloat16 rhs) {
    return __hmax_nan(lhs, rhs);
}

__device__ __forceinline__ __nv_bfloat16 powg(
    __nv_bfloat16 lhs,
    __nv_bfloat16 rhs) {
    return __float2bfloat16(powf(__bfloat162float(lhs), __bfloat162float(rhs)));
}
__device__ __forceinline__ bool isnang(__nv_bfloat16 value) {
    return __hisnan(value);
}
__device__ __forceinline__ __nv_bfloat16 sqrtg(__nv_bfloat16 value) {
    return hsqrt(value);
}
__device__ __forceinline__ __nv_bfloat16 cosg(__nv_bfloat16 value) {
    return hcos(value);
}
__device__ __forceinline__ __nv_bfloat16 sing(__nv_bfloat16 value) {
    return hsin(value);
}
__device__ __forceinline__ __nv_bfloat16 recipg(__nv_bfloat16 value) {
    return __nv_bfloat16(1.0f) / value;
}
__device__ __forceinline__ __nv_bfloat16 tanhg(__nv_bfloat16 value) {
    return __float2bfloat16(tanhf(__bfloat162float(value)));
}
__device__ __forceinline__ __nv_bfloat16 erfg(__nv_bfloat16 value) {
    return __float2bfloat16(erff(__bfloat162float(value)));
}
__device__ __forceinline__ __nv_bfloat16 ceilg(__nv_bfloat16 value) {
    return __float2bfloat16(ceilf(__bfloat162float(value)));
}
__device__ __forceinline__ __nv_bfloat16 floorg(__nv_bfloat16 value) {
    return __float2bfloat16(floorf(__bfloat162float(value)));
}
__device__ __forceinline__ __nv_bfloat16 roundg(__nv_bfloat16 value) {
    return __float2bfloat16(roundf(__bfloat162float(value)));
}
__device__ __forceinline__ __nv_bfloat16 normcdfg(__nv_bfloat16 value) {
    return __float2bfloat16(normcdff(__bfloat162float(value)));
}
__device__ __forceinline__ __nv_bfloat16 logg(__nv_bfloat16 value) {
    return hlog(value);
}
__device__ __forceinline__ __nv_bfloat16 expg(__nv_bfloat16 value) {
    return hexp(value);
}
__device__ __forceinline__ __nv_bfloat16 absg(__nv_bfloat16 value) {
    return __habs(value);
}
__device__ __forceinline__ __nv_bfloat16 copysigng(
    __nv_bfloat16 lhs,
    __nv_bfloat16 rhs) {
    return __float2bfloat16(copysignf(__bfloat162float(lhs), __bfloat162float(rhs)));
}
#endif
