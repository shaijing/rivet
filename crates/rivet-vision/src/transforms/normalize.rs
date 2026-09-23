use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Device, ExactOutput, Tensor};

fn affine_params(mean: &[f32], std: &[f32], channel_count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut scale = Vec::with_capacity(channel_count);
    let mut bias = Vec::with_capacity(channel_count);
    for channel in 0..channel_count {
        let stats_index = if mean.len() == 1 { 0 } else { channel };
        scale.push(1.0 / (255.0 * std[stats_index]));
        bias.push(-mean[stats_index] / std[stats_index]);
    }
    (scale, bias)
}

#[inline(always)]
fn apply_affine(value: u8, scale: &[f32], bias: &[f32], channel: usize) -> f32 {
    value as f32 * scale[channel] + bias[channel]
}

fn write_normalize_contiguous_image(
    values: &[u8],
    dims: &[usize],
    axis_order: ImageAxisOrder,
    scale: &[f32],
    bias: &[f32],
    output: &mut ExactOutput<f32>,
) -> rivet_core::Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    let channel_count = match axis_order {
        ImageAxisOrder::Hwc => dims[2],
        ImageAxisOrder::Chw => dims[0],
    };
    match axis_order {
        ImageAxisOrder::Hwc if channel_count == 3 => {
            for pixels in values.chunks_exact(3) {
                output.write_next(apply_affine(pixels[0], scale, bias, 0))?;
                output.write_next(apply_affine(pixels[1], scale, bias, 1))?;
                output.write_next(apply_affine(pixels[2], scale, bias, 2))?;
            }
            debug_assert!(values.chunks_exact(3).remainder().is_empty());
        }
        ImageAxisOrder::Hwc => {
            for (index, &value) in values.iter().enumerate() {
                output.write_next(apply_affine(value, scale, bias, index % channel_count))?;
            }
        }
        ImageAxisOrder::Chw => {
            let spatial_size = dims[1] * dims[2];
            for channel in 0..channel_count {
                let start = channel * spatial_size;
                let end = start + spatial_size;
                for &value in &values[start..end] {
                    output.write_next(apply_affine(value, scale, bias, channel))?;
                }
            }
        }
    }
    Ok(())
}

fn write_normalize_contiguous_batch(
    values: &[u8],
    dims: &[usize],
    axis_order: ImageAxisOrder,
    scale: &[f32],
    bias: &[f32],
    output: &mut ExactOutput<f32>,
) -> rivet_core::Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    let (batch, channels, spatial_size) = match axis_order {
        ImageAxisOrder::Hwc => (dims[0], dims[3], dims[1] * dims[2]),
        ImageAxisOrder::Chw => (dims[0], dims[1], dims[2] * dims[3]),
    };
    match axis_order {
        ImageAxisOrder::Hwc if channels == 3 => {
            for pixels in values.chunks_exact(3) {
                output.write_next(apply_affine(pixels[0], scale, bias, 0))?;
                output.write_next(apply_affine(pixels[1], scale, bias, 1))?;
                output.write_next(apply_affine(pixels[2], scale, bias, 2))?;
            }
            debug_assert!(values.chunks_exact(3).remainder().is_empty());
        }
        ImageAxisOrder::Hwc => {
            for (index, &value) in values.iter().enumerate() {
                output.write_next(apply_affine(value, scale, bias, index % channels))?;
            }
        }
        ImageAxisOrder::Chw => {
            for batch_index in 0..batch {
                let batch_start = batch_index * channels * spatial_size;
                for channel in 0..channels {
                    let start = batch_start + channel * spatial_size;
                    let end = start + spatial_size;
                    for &value in &values[start..end] {
                        output.write_next(apply_affine(value, scale, bias, channel))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_normalize_contiguous_nhwc_to_nchw(
    values: &[u8],
    dims: &[usize; 4],
    scale: &[f32],
    bias: &[f32],
    output: &mut ExactOutput<f32>,
) -> rivet_core::Result<()> {
    let [batch, height, width, channels] = *dims;
    let spatial_size = height * width;
    let image_size = spatial_size * channels;
    for batch_index in 0..batch {
        let source = &values[batch_index * image_size..(batch_index + 1) * image_size];
        for channel in 0..channels {
            for spatial_index in 0..spatial_size {
                output.write_next(apply_affine(
                    source[spatial_index * channels + channel],
                    scale,
                    bias,
                    channel,
                ))?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizeConfig {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl NormalizeConfig {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.mean.is_empty() || self.std.is_empty() {
            return Err(invalid_argument("normalize mean and std must not be empty"));
        }
        if self.mean.len() != self.std.len() {
            return Err(invalid_argument(
                "normalize mean and std must have the same length",
            ));
        }
        if self.std.iter().any(|value| *value == 0.0) {
            return Err(invalid_argument("normalize std values must be non-zero"));
        }

        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn expanded_affine(
        &self,
        channel_count: usize,
    ) -> RivetResult<(Vec<f32>, Vec<f32>)> {
        self.validate()?;
        if self.mean.len() != 1 && self.mean.len() != channel_count {
            return Err(invalid_argument(format!(
                "normalize mean/std length must be 1 or channel count {channel_count}, got {}",
                self.mean.len()
            )));
        }
        Ok(affine_params(&self.mean, &self.std, channel_count))
    }

    pub fn apply(&self, sample: ImageSample, layout: ImageAxisOrder) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let dims = sample.image.dims();
        if dims.len() != 3 {
            return Err(invalid_shape(format!(
                "normalize requires a rank-3 image, got shape {:?}",
                dims
            )));
        }
        let channel_count = match layout {
            ImageAxisOrder::Hwc => dims[2],
            ImageAxisOrder::Chw => dims[0],
        };
        if self.mean.len() != 1 && self.mean.len() != channel_count {
            return Err(invalid_argument(format!(
                "normalize mean/std length must be 1 or channel count {}, got {}",
                channel_count,
                self.mean.len()
            )));
        }

        let values = if sample.image.dtype() == DType::U8 {
            let image = normalize_u8_to_f32(&sample.image, &self.mean, &self.std, layout)?;
            return Ok(ImageSample::Decoded(DecodedSample {
                image,
                label: sample.label,
            }));
        } else if sample.image.dtype() == DType::F32 {
            sample.image
        } else {
            return Err(invalid_argument(format!(
                "normalize supports uint8 or float32 input, got {:?}",
                sample.image.dtype()
            )));
        };

        let stats_shape = match layout {
            ImageAxisOrder::Hwc => [1, 1, self.mean.len()],
            ImageAxisOrder::Chw => [self.mean.len(), 1, 1],
        };
        let mean = Tensor::from_vec(self.mean.clone(), stats_shape, &Device::Cpu)?;
        let std = Tensor::from_vec(self.std.clone(), stats_shape, &Device::Cpu)?;
        let image = values.broadcast_sub(&mean)?.broadcast_div(&std)?;

        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: sample.label,
        }))
    }

    /// Apply sample normalization after the compiler has validated the
    /// operation and input state.
    pub(crate) fn apply_trusted(
        &self,
        sample: ImageSample,
        layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        debug_assert_eq!(sample.image.rank(), 3);
        let channel_count = match layout {
            ImageAxisOrder::Hwc => sample.image.dims()[2],
            ImageAxisOrder::Chw => sample.image.dims()[0],
        };
        debug_assert!(self.mean.len() == 1 || self.mean.len() == channel_count);

        let values = if sample.image.dtype() == DType::U8 {
            let image = normalize_u8_to_f32_trusted(&sample.image, &self.mean, &self.std, layout)?;
            return Ok(ImageSample::Decoded(DecodedSample {
                image,
                label: sample.label,
            }));
        } else if sample.image.dtype() == DType::F32 {
            sample.image
        } else {
            return Err(invalid_argument(format!(
                "normalize supports uint8 or float32 input, got {:?}",
                sample.image.dtype()
            )));
        };

        let stats_shape = match layout {
            ImageAxisOrder::Hwc => [1, 1, self.mean.len()],
            ImageAxisOrder::Chw => [self.mean.len(), 1, 1],
        };
        let mean = Tensor::from_vec(self.mean.clone(), stats_shape, &Device::Cpu)?;
        let std = Tensor::from_vec(self.std.clone(), stats_shape, &Device::Cpu)?;
        let image = values.broadcast_sub(&mean)?.broadcast_div(&std)?;

        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: sample.label,
        }))
    }

    /// Apply normalization to a stacked rank-4 image batch.
    pub fn apply_batch(&self, input: Tensor, layout: ImageAxisOrder) -> RivetResult<Tensor> {
        self.validate_batch_input(&input, layout)?;

        let values = match input.dtype() {
            DType::U8 => {
                return normalize_u8_batch_to_f32(&input, &self.mean, &self.std, layout);
            }
            DType::F32 => input,
            dtype => {
                return Err(invalid_argument(format!(
                    "normalize supports uint8 or float32 input, got {:?}",
                    dtype
                )));
            }
        };

        let stats_shape = match layout {
            ImageAxisOrder::Hwc => vec![1, 1, 1, self.mean.len()],
            ImageAxisOrder::Chw => vec![1, self.mean.len(), 1, 1],
        };
        let mean = Tensor::from_vec(self.mean.clone(), stats_shape.clone(), &Device::Cpu)?;
        let std = Tensor::from_vec(self.std.clone(), stats_shape, &Device::Cpu)?;
        Ok(values.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    /// Apply a batch normalization whose configuration and input state were
    /// checked by pipeline compilation.
    pub(crate) fn apply_batch_trusted(
        &self,
        input: Tensor,
        layout: ImageAxisOrder,
    ) -> RivetResult<Tensor> {
        debug_assert!(!self.mean.is_empty());
        debug_assert_eq!(self.mean.len(), self.std.len());
        debug_assert!(self.std.iter().all(|value| *value != 0.0));
        debug_assert_eq!(input.rank(), 4);

        let values = match input.dtype() {
            DType::U8 => {
                return normalize_u8_batch_to_f32_trusted(&input, &self.mean, &self.std, layout);
            }
            DType::F32 => input,
            dtype => {
                return Err(invalid_argument(format!(
                    "normalize supports uint8 or float32 input, got {:?}",
                    dtype
                )));
            }
        };

        let stats_shape = match layout {
            ImageAxisOrder::Hwc => vec![1, 1, 1, self.mean.len()],
            ImageAxisOrder::Chw => vec![1, self.mean.len(), 1, 1],
        };
        let mean = Tensor::from_vec(self.mean.clone(), stats_shape.clone(), &Device::Cpu)?;
        let std = Tensor::from_vec(self.std.clone(), stats_shape, &Device::Cpu)?;
        Ok(values.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    /// Fused `Normalize` + `HWC -> CHW` batch operation.
    ///
    /// This is intentionally limited to uint8 HWC input. The compiler only
    /// emits it for that exact adjacent operation pair, so unsupported input
    /// states retain the ordinary normalize/layout path.
    pub fn apply_batch_to_chw(&self, input: Tensor) -> RivetResult<Tensor> {
        self.validate_batch_input(&input, ImageAxisOrder::Hwc)?;
        if input.dtype() != DType::U8 {
            return Err(invalid_argument(format!(
                "normalize_to_chw requires uint8 input, got {:?}",
                input.dtype()
            )));
        }

        normalize_u8_batch_to_nchw_f32(&input, &self.mean, &self.std)
    }

    /// Apply the fused path after compilation has established U8 NHWC input.
    pub(crate) fn apply_batch_to_chw_trusted(&self, input: Tensor) -> RivetResult<Tensor> {
        debug_assert!(!self.mean.is_empty());
        debug_assert_eq!(self.mean.len(), self.std.len());
        debug_assert!(self.std.iter().all(|value| *value != 0.0));
        debug_assert_eq!(input.rank(), 4);
        debug_assert_eq!(input.dtype(), DType::U8);
        normalize_u8_batch_to_nchw_f32_trusted(&input, &self.mean, &self.std)
    }

    fn validate_batch_input(&self, input: &Tensor, layout: ImageAxisOrder) -> RivetResult<()> {
        if input.rank() != 4 {
            return Err(invalid_shape(format!(
                "batch normalize requires a rank-4 image batch, got shape {:?}",
                input.dims()
            )));
        }
        if self.mean.is_empty() || self.std.is_empty() {
            return Err(invalid_argument("normalize mean and std must not be empty"));
        }
        if self.mean.len() != self.std.len() {
            return Err(invalid_argument(
                "normalize mean and std must have the same length",
            ));
        }
        if self.std.iter().any(|value| *value == 0.0) {
            return Err(invalid_argument("normalize std values must be non-zero"));
        }

        let channel_count = match layout {
            ImageAxisOrder::Hwc => input.dims()[3],
            ImageAxisOrder::Chw => input.dims()[1],
        };
        if self.mean.len() != 1 && self.mean.len() != channel_count {
            return Err(invalid_argument(format!(
                "normalize mean/std length must be 1 or channel count {}, got {}",
                channel_count,
                self.mean.len()
            )));
        }

        Ok(())
    }
}

/// Fused uint8 image-batch normalization. The input is read in logical tensor
/// order and the output is allocated exactly once:
/// `((input as f32) / 255.0 - mean[channel]) / std[channel]`.
///
/// `axis_order` describes the last three dimensions, so the accepted shapes
/// are `[N, H, W, C]` for HWC and `[N, C, H, W]` for CHW. The layout supplied
/// by the tensor is always used when reading storage, which keeps this path
/// correct for non-contiguous batch views.
pub fn normalize_u8_batch_to_f32(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageAxisOrder,
) -> RivetResult<Tensor> {
    normalize_u8_batch_to_f32_impl(input, mean, std, axis_order, true)
}

pub(crate) fn normalize_u8_batch_to_f32_trusted(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageAxisOrder,
) -> RivetResult<Tensor> {
    normalize_u8_batch_to_f32_impl(input, mean, std, axis_order, false)
}

fn normalize_u8_batch_to_f32_impl(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageAxisOrder,
    validate: bool,
) -> RivetResult<Tensor> {
    if validate {
        if input.dtype() != DType::U8 {
            return Err(invalid_argument(format!(
                "normalize_u8_batch_to_f32 requires uint8 input, got {:?}",
                input.dtype()
            )));
        }
        if input.rank() != 4 {
            return Err(invalid_shape(format!(
                "normalize_u8_batch_to_f32 requires a rank-4 image batch, got shape {:?}",
                input.dims()
            )));
        }
        if mean.is_empty() || std.is_empty() {
            return Err(invalid_argument("normalize mean and std must not be empty"));
        }
        if mean.len() != std.len() {
            return Err(invalid_argument(
                "normalize mean and std must have the same length",
            ));
        }
        if std.iter().any(|value| *value == 0.0) {
            return Err(invalid_argument("normalize std values must be non-zero"));
        }
    } else {
        debug_assert_eq!(input.dtype(), DType::U8);
        debug_assert_eq!(input.rank(), 4);
        debug_assert!(!mean.is_empty());
        debug_assert_eq!(mean.len(), std.len());
        debug_assert!(std.iter().all(|value| *value != 0.0));
    }

    let channel_count = match axis_order {
        ImageAxisOrder::Hwc => input.dims()[3],
        ImageAxisOrder::Chw => input.dims()[1],
    };
    if validate && mean.len() != 1 && mean.len() != channel_count {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channel_count,
            mean.len()
        )));
    }
    debug_assert!(mean.len() == 1 || mean.len() == channel_count);

    let (scale, bias) = affine_params(mean, std, channel_count);
    let dims = input.dims().to_vec();
    let device = input.device().clone();

    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };

        if let Some((start, end)) = layout.contiguous_offsets() {
            let values = values
                .get(start..end)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            return Tensor::from_exact_writer::<f32, _, _>(dims, &device, |output| {
                write_normalize_contiguous_batch(
                    values,
                    &input.dims(),
                    axis_order,
                    &scale,
                    &bias,
                    output,
                )
            })
            .map_err(Into::into);
        }

        let [_, dim1, dim2, dim3] = dims.as_slice() else {
            return Err(rivet_core::Error::InvalidRank {
                expected: 4,
                actual: dims.len(),
            });
        };
        let (dim1, dim2, dim3) = (*dim1, *dim2, *dim3);
        let output = layout
            .strided_index()
            .enumerate()
            .map(|(logical_index, physical_index)| {
                let value = *values
                    .get(physical_index)
                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                let channel = match axis_order {
                    ImageAxisOrder::Hwc => logical_index % dim3,
                    ImageAxisOrder::Chw => (logical_index % (dim1 * dim2 * dim3)) / (dim2 * dim3),
                };
                Ok(apply_affine(value, &scale, &bias, channel))
            });

        Tensor::from_exact_try_iter(output, dims, &device).map_err(Into::into)
    })?)
}

/// Fused uint8 NHWC normalization with direct NCHW F32 output.
///
/// The source is read in its logical HWC order while the output is written in
/// NCHW order, so no intermediate NHWC F32 tensor or layout view is created.
/// The tensor layout is consulted for every read, which also supports HWC
/// views with non-standard strides and offsets.
pub fn normalize_u8_batch_to_nchw_f32(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
) -> RivetResult<Tensor> {
    normalize_u8_batch_to_nchw_f32_impl(input, mean, std, true)
}

#[cfg(feature = "cuda")]
pub(crate) fn normalize_u8_batch_to_nchw_f32_cuda(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
) -> RivetResult<Tensor> {
    if input.dtype() != DType::U8 {
        return Err(invalid_argument(format!(
            "normalize_u8_batch_to_nchw_f32_cuda requires uint8 input, got {:?}",
            input.dtype()
        )));
    }
    if input.dims().len() != 4 {
        return Err(invalid_shape(format!(
            "normalize_u8_batch_to_nchw_f32_cuda requires a rank-4 image batch, got shape {:?}",
            input.dims()
        )));
    }
    let channels = input.dims()[3];
    if mean.is_empty() || mean.len() != std.len() || (mean.len() != 1 && mean.len() != channels) {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {channels}, got {} and {}",
            mean.len(),
            std.len()
        )));
    }
    if std.iter().any(|value| *value == 0.0) {
        return Err(invalid_argument("normalize std values must be non-zero"));
    }
    let (scale, bias) = affine_params(mean, std, channels);
    input
        .cuda_normalize_u8_nhwc_to_nchw_f32(&scale, &bias)
        .map_err(Into::into)
}

pub(crate) fn normalize_u8_batch_to_nchw_f32_trusted(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
) -> RivetResult<Tensor> {
    normalize_u8_batch_to_nchw_f32_impl(input, mean, std, false)
}

fn normalize_u8_batch_to_nchw_f32_impl(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    validate: bool,
) -> RivetResult<Tensor> {
    if validate {
        if input.dtype() != DType::U8 {
            return Err(invalid_argument(format!(
                "normalize_u8_batch_to_nchw_f32 requires uint8 input, got {:?}",
                input.dtype()
            )));
        }
        if input.rank() != 4 {
            return Err(invalid_shape(format!(
                "normalize_u8_batch_to_nchw_f32 requires a rank-4 image batch, got shape {:?}",
                input.dims()
            )));
        }
        if mean.is_empty() || std.is_empty() {
            return Err(invalid_argument("normalize mean and std must not be empty"));
        }
        if mean.len() != std.len() {
            return Err(invalid_argument(
                "normalize mean and std must have the same length",
            ));
        }
        if std.iter().any(|value| *value == 0.0) {
            return Err(invalid_argument("normalize std values must be non-zero"));
        }
    } else {
        debug_assert_eq!(input.dtype(), DType::U8);
        debug_assert_eq!(input.rank(), 4);
        debug_assert!(!mean.is_empty());
        debug_assert_eq!(mean.len(), std.len());
        debug_assert!(std.iter().all(|value| *value != 0.0));
    }

    let [batch, height, width, channels] = input.dims() else {
        return Err(invalid_shape(format!(
            "normalize_u8_batch_to_nchw_f32 requires NHWC input, got shape {:?}",
            input.dims()
        )));
    };
    if validate && mean.len() != 1 && mean.len() != *channels {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channels,
            mean.len()
        )));
    }
    debug_assert!(mean.len() == 1 || mean.len() == *channels);

    let (scale, bias) = affine_params(mean, std, *channels);
    let dims = [*batch, *channels, *height, *width];
    let device = input.device().clone();
    let elem_count = input.elem_count();
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };

        if let Some((start, end)) = layout.contiguous_offsets() {
            let values = values
                .get(start..end)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            let source_dims = [*batch, *height, *width, *channels];
            return Tensor::from_exact_writer::<f32, _, _>(dims, &device, |output| {
                write_normalize_contiguous_nhwc_to_nchw(values, &source_dims, &scale, &bias, output)
            })
            .map_err(Into::into);
        }

        let stride = layout.stride();
        let output = (0..elem_count).map(|output_index| {
            let batch_index = output_index / (*channels * *height * *width);
            let channel = (output_index / (*height * *width)) % *channels;
            let height_index = (output_index / *width) % *height;
            let width_index = output_index % *width;
            let physical_offset = batch_index
                .checked_mul(stride[0])
                .and_then(|offset| offset.checked_add(height_index.checked_mul(stride[1])?))
                .and_then(|offset| offset.checked_add(width_index.checked_mul(stride[2])?))
                .and_then(|offset| offset.checked_add(channel.checked_mul(stride[3])?))
                .and_then(|offset| layout.start_offset().checked_add(offset))
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            let value = *values
                .get(physical_offset)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            Ok(apply_affine(value, &scale, &bias, channel))
        });

        Tensor::from_exact_try_iter(output, dims, &device).map_err(Into::into)
    })?)
}

/// Fused uint8 image normalization. The input is read in logical tensor
/// order and the output is allocated exactly once:
/// `((input as f32) / 255.0 - mean[channel]) / std[channel]`.
pub fn normalize_u8_to_f32(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageAxisOrder,
) -> RivetResult<Tensor> {
    normalize_u8_to_f32_impl(input, mean, std, axis_order, true)
}

pub(crate) fn normalize_u8_to_f32_trusted(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageAxisOrder,
) -> RivetResult<Tensor> {
    normalize_u8_to_f32_impl(input, mean, std, axis_order, false)
}

fn normalize_u8_to_f32_impl(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageAxisOrder,
    validate: bool,
) -> RivetResult<Tensor> {
    if validate {
        if input.dtype() != DType::U8 {
            return Err(invalid_argument(format!(
                "normalize_u8_to_f32 requires uint8 input, got {:?}",
                input.dtype()
            )));
        }
        if input.rank() != 3 {
            return Err(invalid_shape(format!(
                "normalize_u8_to_f32 requires a rank-3 image, got shape {:?}",
                input.dims()
            )));
        }
        if mean.is_empty() || std.is_empty() {
            return Err(invalid_argument("normalize mean and std must not be empty"));
        }
        if mean.len() != std.len() {
            return Err(invalid_argument(
                "normalize mean and std must have the same length",
            ));
        }
        if std.iter().any(|value| *value == 0.0) {
            return Err(invalid_argument("normalize std values must be non-zero"));
        }
    } else {
        debug_assert_eq!(input.dtype(), DType::U8);
        debug_assert_eq!(input.rank(), 3);
        debug_assert!(!mean.is_empty());
        debug_assert_eq!(mean.len(), std.len());
        debug_assert!(std.iter().all(|value| *value != 0.0));
    }

    let channel_count = match axis_order {
        ImageAxisOrder::Hwc => input.dims()[2],
        ImageAxisOrder::Chw => input.dims()[0],
    };
    if validate && mean.len() != 1 && mean.len() != channel_count {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channel_count,
            mean.len()
        )));
    }
    debug_assert!(mean.len() == 1 || mean.len() == channel_count);

    let spatial_size = match axis_order {
        ImageAxisOrder::Hwc => 1,
        ImageAxisOrder::Chw => input.dims()[1]
            .checked_mul(input.dims()[2])
            .ok_or_else(|| invalid_shape("normalize image dimensions overflow"))?,
    };
    let (scale, bias) = affine_params(mean, std, channel_count);
    let dims = input.dims().to_vec();
    let device = input.device().clone();
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };

        if let Some((start, end)) = layout.contiguous_offsets() {
            let values = values
                .get(start..end)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            return Tensor::from_exact_writer::<f32, _, _>(dims, &device, |output| {
                write_normalize_contiguous_image(
                    values,
                    &input.dims(),
                    axis_order,
                    &scale,
                    &bias,
                    output,
                )
            })
            .map_err(Into::into);
        }

        let output = layout
            .strided_index()
            .enumerate()
            .map(|(logical_index, physical_index)| {
                let value = *values
                    .get(physical_index)
                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                let channel = match axis_order {
                    ImageAxisOrder::Hwc => logical_index % channel_count,
                    ImageAxisOrder::Chw => logical_index / spatial_size,
                };
                Ok(apply_affine(value, &scale, &bias, channel))
            });

        Tensor::from_exact_try_iter(output, dims, &device).map_err(Into::into)
    })?)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::normalize_u8_batch_to_nchw_f32_cuda;
    use super::{
        NormalizeConfig, normalize_u8_batch_to_f32, normalize_u8_batch_to_nchw_f32,
        normalize_u8_to_f32,
    };
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn normalize_config_uses_fused_u8_path() {
        let image = Tensor::from_vec(vec![0u8, 255, 128], [1, 1, 3], &Device::Cpu).unwrap();
        let sample = ImageSample::Decoded(DecodedSample { image, label: 0 });
        let out = NormalizeConfig::new(vec![0.5], vec![0.5])
            .apply(sample, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();

        assert_eq!(out.image.dtype(), DType::F32);
        let values = out.image.to_vec::<f32>().unwrap();
        assert_eq!(values[0], -1.0);
        assert_eq!(values[1], 1.0);
    }

    #[test]
    fn fused_normalize_supports_per_channel_chw_views() {
        let hwc =
            Tensor::from_vec(vec![0u8, 64, 128, 255, 32, 96], [1, 2, 3], &Device::Cpu).unwrap();
        let chw = hwc.permute(&[2, 0, 1]).unwrap();
        let out = normalize_u8_to_f32(
            &chw,
            &[0.0, 0.5, 1.0],
            &[1.0, 0.5, 0.25],
            ImageAxisOrder::Chw,
        )
        .unwrap();

        assert_eq!(out.dims(), [3, 1, 2]);
        let values = out.to_vec::<f32>().unwrap();
        let expected = [0.0, 1.0, -0.49803922, -0.7490196, -1.9921569, -2.4941177];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn fused_normalize_supports_contiguous_chw_input() {
        let chw =
            Tensor::from_vec(vec![0u8, 255, 64, 32, 128, 96], [3, 1, 2], &Device::Cpu).unwrap();
        let out = normalize_u8_to_f32(
            &chw,
            &[0.0, 0.5, 1.0],
            &[1.0, 0.5, 0.25],
            ImageAxisOrder::Chw,
        )
        .unwrap();

        let expected = [0.0, 1.0, -0.49803922, -0.7490196, -1.9921569, -2.4941177];
        for (actual, expected) in out.to_vec::<f32>().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn batch_normalize_supports_nhwc_and_chw() {
        let input = Tensor::from_vec(
            (0..12).map(|value| value as u8).collect(),
            [2, 1, 2, 3],
            &Device::Cpu,
        )
        .unwrap();
        let config = NormalizeConfig::new(vec![0.0, 0.5, 1.0], vec![1.0, 0.5, 0.25]);

        let nhwc = config
            .apply_batch(input.clone(), ImageAxisOrder::Hwc)
            .unwrap();
        assert_eq!(nhwc.dims(), [2, 1, 2, 3]);
        assert_batch_values(&input, &nhwc, ImageAxisOrder::Hwc, &config);

        let nchw_input = input.permute(&[0, 3, 1, 2]).unwrap();
        assert!(!nchw_input.is_contiguous());
        let nchw = config
            .apply_batch(nchw_input.clone(), ImageAxisOrder::Chw)
            .unwrap();
        assert_eq!(nchw.dims(), [2, 3, 1, 2]);
        assert_batch_values(&nchw_input, &nchw, ImageAxisOrder::Chw, &config);

        let contiguous_nchw = Tensor::from_vec(
            (0..12).map(|value| value as u8).collect(),
            [2, 3, 1, 2],
            &Device::Cpu,
        )
        .unwrap();
        let contiguous_output = config
            .apply_batch(contiguous_nchw.clone(), ImageAxisOrder::Chw)
            .unwrap();
        assert_batch_values(
            &contiguous_nchw,
            &contiguous_output,
            ImageAxisOrder::Chw,
            &config,
        );
    }

    fn assert_batch_values(
        input: &Tensor,
        output: &Tensor,
        layout: ImageAxisOrder,
        config: &NormalizeConfig,
    ) {
        let source = input.to_vec::<u8>().unwrap();
        let actual = output.to_vec::<f32>().unwrap();
        let spatial_size = match layout {
            ImageAxisOrder::Hwc => input.dims()[1] * input.dims()[2],
            ImageAxisOrder::Chw => input.dims()[2] * input.dims()[3],
        };
        for (logical_index, (&value, &actual)) in source.iter().zip(&actual).enumerate() {
            let channel = match layout {
                ImageAxisOrder::Hwc => logical_index % input.dims()[3],
                ImageAxisOrder::Chw => (logical_index / spatial_size) % input.dims()[1],
            };
            let expected = (value as f32 / 255.0 - config.mean[channel]) / config.std[channel];
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn batch_fused_normalize_supports_scalar_statistics() {
        let input = Tensor::from_vec(vec![0u8, 255], [1, 1, 2, 1], &Device::Cpu).unwrap();
        let output =
            normalize_u8_batch_to_f32(&input, &[0.5], &[0.5], ImageAxisOrder::Hwc).unwrap();

        assert_eq!(output.dims(), [1, 1, 2, 1]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.to_vec::<f32>().unwrap(), [-1.0, 1.0]);
    }

    #[test]
    fn batch_fused_normalize_preserves_empty_shape() {
        let input = Tensor::from_vec(Vec::<u8>::new(), [0, 2, 2, 3], &Device::Cpu).unwrap();
        let output = normalize_u8_batch_to_f32(
            &input,
            &[0.0, 0.5, 1.0],
            &[1.0, 0.5, 0.25],
            ImageAxisOrder::Hwc,
        )
        .unwrap();

        assert_eq!(output.dims(), [0, 2, 2, 3]);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.elem_count(), 0);
    }

    #[test]
    fn batch_fused_normalize_rejects_non_u8_input() {
        let input = Tensor::from_vec(vec![0.0f32; 3], [1, 1, 1, 3], &Device::Cpu).unwrap();
        let err = normalize_u8_batch_to_f32(&input, &[0.0; 3], &[1.0; 3], ImageAxisOrder::Hwc)
            .unwrap_err();
        assert!(
            err.to_string().contains("requires uint8 input"),
            "got: {err}"
        );
    }

    #[test]
    fn fused_normalize_to_nchw_matches_normalize_then_layout() {
        let base = Tensor::from_vec(
            (0..24).map(|value| value as u8).collect(),
            [2, 2, 3, 2],
            &Device::Cpu,
        )
        .unwrap();
        let input = base.permute(&[0, 1, 3, 2]).unwrap();
        assert!(!input.is_contiguous());
        let config = NormalizeConfig::new(vec![0.0, 0.5, 1.0], vec![1.0, 0.5, 0.25]);

        let expected = config
            .apply_batch(input.clone(), ImageAxisOrder::Hwc)
            .unwrap()
            .permute(&[0, 3, 1, 2])
            .unwrap();
        let actual = normalize_u8_batch_to_nchw_f32(&input, &config.mean, &config.std).unwrap();

        assert_eq!(actual.dims(), [2, 3, 2, 2]);
        for (actual, expected) in actual
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(expected.to_vec::<f32>().unwrap())
        {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_normalize_layout_matches_cpu_for_multi_image_rgb_batch() {
        let Ok(device) = Device::cuda(0) else {
            eprintln!("skipping CUDA normalize parity test because device 0 is unavailable");
            return;
        };
        let input = Tensor::from_vec(
            (0..2 * 2 * 3 * 3).map(|value| (value * 7) as u8).collect(),
            [2, 2, 3, 3],
            &Device::Cpu,
        )
        .unwrap();
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let expected = normalize_u8_batch_to_nchw_f32(&input, &mean, &std).unwrap();
        let input = input.to_device(&device).unwrap();
        let actual = normalize_u8_batch_to_nchw_f32_cuda(&input, &mean, &std).unwrap();

        assert_eq!(actual.dims(), [2, 3, 2, 3]);
        assert_eq!(actual.dtype(), DType::F32);
        assert!(actual.device().is_cuda());
        let expected = expected.to_vec::<f32>().unwrap();
        let actual = actual.to_vec::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }
}
