#include <stddef.h>
#include <stdint.h>

#include "cuda_utils.cuh"

extern "C" __global__ void normalize_nhwc_u8_to_nchw_f32(
    const uint8_t *input,
    float *output,
    size_t batch,
    size_t height,
    size_t width,
    size_t channels,
    const float *scale,
    const float *bias) {
    const size_t spatial = height * width;
    const size_t output_len = batch * channels * spatial;
    for (uint64_t output_index = rivet_global_index(); output_index < output_len;
         output_index += rivet_global_stride()) {
        const size_t channel = (output_index / spatial) % channels;
        const size_t batch_index = output_index / (channels * spatial);
        const size_t spatial_index = output_index % spatial;
        const size_t input_index = (batch_index * spatial + spatial_index) * channels + channel;
        output[output_index] = static_cast<float>(input[input_index]) * scale[channel] + bias[channel];
    }
}

// Common RGB path: each thread loads one contiguous NHWC pixel and writes its
// three normalized channels to contiguous NCHW planes.
extern "C" __global__ void normalize_nhwc3_u8_to_nchw_f32(
    const uint8_t *input,
    float *output,
    size_t batch,
    size_t height,
    size_t width,
    const float *scale,
    const float *bias) {
    const size_t spatial = height * width;
    const size_t pixels = batch * spatial;
    for (uint64_t pixel_index = rivet_global_index(); pixel_index < pixels;
         pixel_index += rivet_global_stride()) {
        const size_t batch_index = pixel_index / spatial;
        const size_t spatial_index = pixel_index % spatial;
        const size_t input_index = pixel_index * 3;
        const size_t output_index = batch_index * 3 * spatial + spatial_index;
        output[output_index] = static_cast<float>(input[input_index]) * scale[0] + bias[0];
        output[output_index + spatial] =
            static_cast<float>(input[input_index + 1]) * scale[1] + bias[1];
        output[output_index + 2 * spatial] =
            static_cast<float>(input[input_index + 2]) * scale[2] + bias[2];
    }
}

__device__ __forceinline__ float vision_sinc(float x) {
    if (x == 0.0f) return 1.0f;
    const float pix = 3.14159265358979323846f * x;
    return sinf(pix) / pix;
}

template <int Filter>
__device__ __forceinline__ float vision_filter_weight(float x) {
    const float a = fabsf(x);
    if constexpr (Filter == 0) {
        return 1.0f; // image::FilterType::Nearest
    } else if constexpr (Filter == 1) {
        return a < 1.0f ? 1.0f - a : 0.0f; // Triangle
    } else if constexpr (Filter == 2) {
        if (a < 1.0f) return (9.0f * a * a * a - 15.0f * a * a + 6.0f) / 6.0f;
        if (a < 2.0f) return (-3.0f * a * a * a + 15.0f * a * a - 24.0f * a + 12.0f) / 6.0f;
        return 0.0f;
    } else {
        if (a < 3.0f) return vision_sinc(x) * vision_sinc(x / 3.0f);
        return 0.0f;
    }
}

template <int Filter>
__device__ __forceinline__ int vision_filter_support() {
    if constexpr (Filter == 0) return 0;
    if constexpr (Filter == 1) return 1;
    if constexpr (Filter == 2) return 2;
    return 3;
}

// Replicates image::imageops::resize's separable filter for RGB U8 images,
// then fuses optional per-sample crop/flip, channel normalization, and HWC to
// CHW materialization. `crop_params` stores [x, y, width, height,
// source_reverse_x, source_reverse_y, output_flip_x, output_flip_y] per sample.
template <int Filter>
__device__ __forceinline__ void vision_augment_normalize_nhwc3_u8_to_nchw_f32_impl(
    const uint8_t *input,
    float *output,
    size_t batch,
    size_t input_height,
    size_t input_width,
    size_t output_height,
    size_t output_width,
    const uint32_t *crop_params,
    const float *scale,
    const float *bias) {
    const size_t output_spatial = output_height * output_width;
    const size_t output_pixels = batch * output_spatial;
    const int support = vision_filter_support<Filter>();
    for (uint64_t pixel_index = rivet_global_index(); pixel_index < output_pixels;
         pixel_index += rivet_global_stride()) {
        const size_t batch_index = pixel_index / output_spatial;
        const size_t spatial_index = pixel_index % output_spatial;
        const size_t out_y = spatial_index / output_width;
        const size_t out_x = spatial_index % output_width;
        const uint32_t *params = crop_params + batch_index * 8;
        const size_t crop_x = params[0];
        const size_t crop_y = params[1];
        const size_t crop_width = params[2];
        const size_t crop_height = params[3];
        const bool source_reverse_x = params[4] != 0;
        const bool source_reverse_y = params[5] != 0;
        const bool output_flip_x = params[6] != 0;
        const bool output_flip_y = params[7] != 0;

        float sample_x;
        float sample_y;
        if (crop_width == output_width) {
            sample_x = static_cast<float>(output_flip_x ? output_width - 1 - out_x : out_x);
        } else {
            const size_t resize_x = output_flip_x ? output_width - 1 - out_x : out_x;
            sample_x = (static_cast<float>(resize_x) + 0.5f) *
                       (static_cast<float>(crop_width) / static_cast<float>(output_width)) - 0.5f;
        }
        if (crop_height == output_height) {
            sample_y = static_cast<float>(output_flip_y ? output_height - 1 - out_y : out_y);
        } else {
            const size_t resize_y = output_flip_y ? output_height - 1 - out_y : out_y;
            sample_y = (static_cast<float>(resize_y) + 0.5f) *
                       (static_cast<float>(crop_height) / static_cast<float>(output_height)) - 0.5f;
        }

        float values[3] = {0.0f, 0.0f, 0.0f};
        if (crop_width == output_width && crop_height == output_height) {
            size_t local_x = static_cast<size_t>(sample_x);
            size_t local_y = static_cast<size_t>(sample_y);
            if (source_reverse_x) local_x = crop_width - 1 - local_x;
            if (source_reverse_y) local_y = crop_height - 1 - local_y;
            const size_t input_index =
                ((batch_index * input_height + crop_y + local_y) * input_width + crop_x + local_x) * 3;
#pragma unroll
            for (int channel = 0; channel < 3; ++channel) {
                values[channel] = static_cast<float>(input[input_index + channel]);
            }
        } else {
            const float ratio_x = static_cast<float>(crop_width) / static_cast<float>(output_width);
            const float ratio_y = static_cast<float>(crop_height) / static_cast<float>(output_height);
            const float scale_x = ratio_x < 1.0f ? 1.0f : ratio_x;
            const float scale_y = ratio_y < 1.0f ? 1.0f : ratio_y;
            const float support_x = static_cast<float>(support) * scale_x;
            const float support_y = static_cast<float>(support) * scale_y;
            int left_x = static_cast<int>(floorf(sample_x + 0.5f - support_x));
            int right_x = static_cast<int>(ceilf(sample_x + 0.5f + support_x));
            int left_y = static_cast<int>(floorf(sample_y + 0.5f - support_y));
            int right_y = static_cast<int>(ceilf(sample_y + 0.5f + support_y));
            left_x = max(0, min(left_x, static_cast<int>(crop_width) - 1));
            left_y = max(0, min(left_y, static_cast<int>(crop_height) - 1));
            right_x = max(left_x + 1, min(right_x, static_cast<int>(crop_width)));
            right_y = max(left_y + 1, min(right_y, static_cast<int>(crop_height)));
            float sum_x = 0.0f;
            for (int ix = left_x; ix < right_x; ++ix) {
                sum_x += vision_filter_weight<Filter>((static_cast<float>(ix) - sample_x) / scale_x);
            }
            float sum_y = 0.0f;
            for (int iy = left_y; iy < right_y; ++iy) {
                sum_y += vision_filter_weight<Filter>((static_cast<float>(iy) - sample_y) / scale_y);
            }
            for (int ix = left_x; ix < right_x; ++ix) {
                const float wx = vision_filter_weight<Filter>((static_cast<float>(ix) - sample_x) / scale_x) / sum_x;
                float vertical[3] = {0.0f, 0.0f, 0.0f};
                for (int iy = left_y; iy < right_y; ++iy) {
                    const float wy = vision_filter_weight<Filter>((static_cast<float>(iy) - sample_y) / scale_y) / sum_y;
                    size_t input_x = crop_x + static_cast<size_t>(ix);
                    size_t input_y = crop_y + static_cast<size_t>(iy);
                    if (source_reverse_x) input_x = crop_x + crop_width - 1 - static_cast<size_t>(ix);
                    if (source_reverse_y) input_y = crop_y + crop_height - 1 - static_cast<size_t>(iy);
                    const size_t input_index =
                        ((batch_index * input_height + input_y) * input_width + input_x) * 3;
#pragma unroll
                    for (int channel = 0; channel < 3; ++channel) {
                        vertical[channel] += static_cast<float>(input[input_index + channel]) * wy;
                    }
                }
#pragma unroll
                for (int channel = 0; channel < 3; ++channel) {
                    values[channel] += vertical[channel] * wx;
                }
            }
#pragma unroll
            for (int channel = 0; channel < 3; ++channel) {
                values[channel] = fminf(255.0f, fmaxf(0.0f, roundf(values[channel])));
            }
        }
#pragma unroll
        for (int channel = 0; channel < 3; ++channel) {
            const size_t output_plane = batch_index * 3 * output_spatial + channel * output_spatial + spatial_index;
            output[output_plane] = values[channel] * scale[channel] + bias[channel];
        }
    }
}

#define VISION_AUGMENT_KERNEL(NAME, FILTER) \
extern "C" __global__ void NAME( \
    const uint8_t *input, float *output, size_t batch, size_t input_height, \
    size_t input_width, size_t output_height, size_t output_width, \
    const uint32_t *crop_params, const float *scale, const float *bias) { \
    vision_augment_normalize_nhwc3_u8_to_nchw_f32_impl<FILTER>( \
        input, output, batch, input_height, input_width, output_height, \
        output_width, crop_params, scale, bias); \
}

VISION_AUGMENT_KERNEL(vision_augment_normalize_nhwc3_u8_to_nchw_f32_nearest, 0)
VISION_AUGMENT_KERNEL(vision_augment_normalize_nhwc3_u8_to_nchw_f32_bilinear, 1)
VISION_AUGMENT_KERNEL(vision_augment_normalize_nhwc3_u8_to_nchw_f32_bicubic, 2)
VISION_AUGMENT_KERNEL(vision_augment_normalize_nhwc3_u8_to_nchw_f32_lanczos3, 3)

#undef VISION_AUGMENT_KERNEL
