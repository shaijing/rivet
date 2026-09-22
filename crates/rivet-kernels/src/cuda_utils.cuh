#pragma once

#include <stddef.h>
#include <stdint.h>

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
