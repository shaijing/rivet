#pragma once

#include <stddef.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t rivet_global_index() {
    return static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint64_t rivet_global_stride() {
    return static_cast<uint64_t>(blockDim.x) * gridDim.x;
}
