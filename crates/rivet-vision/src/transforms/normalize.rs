use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Device, Tensor};

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

fn normalize_contiguous_image(
    values: &[u8],
    dims: &[usize],
    axis_order: ImageAxisOrder,
    scale: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = Vec::with_capacity(values.len());
    if values.is_empty() {
        return output;
    }

    let channel_count = match axis_order {
        ImageAxisOrder::Hwc => dims[2],
        ImageAxisOrder::Chw => dims[0],
    };
    match axis_order {
        ImageAxisOrder::Hwc if channel_count == 3 => {
            for pixels in values.chunks_exact(3) {
                output.push(apply_affine(pixels[0], scale, bias, 0));
                output.push(apply_affine(pixels[1], scale, bias, 1));
                output.push(apply_affine(pixels[2], scale, bias, 2));
            }
            debug_assert!(values.chunks_exact(3).remainder().is_empty());
        }
        ImageAxisOrder::Hwc => {
            for (index, &value) in values.iter().enumerate() {
                output.push(apply_affine(value, scale, bias, index % channel_count));
            }
        }
        ImageAxisOrder::Chw => {
            let spatial_size = dims[1] * dims[2];
            for channel in 0..channel_count {
                let start = channel * spatial_size;
                let end = start + spatial_size;
                output.extend(
                    values[start..end]
                        .iter()
                        .map(|&value| apply_affine(value, scale, bias, channel)),
                );
            }
        }
    }
    output
}

fn normalize_contiguous_batch(
    values: &[u8],
    dims: &[usize],
    axis_order: ImageAxisOrder,
    scale: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = Vec::with_capacity(values.len());
    if values.is_empty() {
        return output;
    }

    let (batch, channels, spatial_size) = match axis_order {
        ImageAxisOrder::Hwc => (dims[0], dims[3], dims[1] * dims[2]),
        ImageAxisOrder::Chw => (dims[0], dims[1], dims[2] * dims[3]),
    };
    match axis_order {
        ImageAxisOrder::Hwc if channels == 3 => {
            for pixels in values.chunks_exact(3) {
                output.push(apply_affine(pixels[0], scale, bias, 0));
                output.push(apply_affine(pixels[1], scale, bias, 1));
                output.push(apply_affine(pixels[2], scale, bias, 2));
            }
            debug_assert!(values.chunks_exact(3).remainder().is_empty());
        }
        ImageAxisOrder::Hwc => {
            for (index, &value) in values.iter().enumerate() {
                output.push(apply_affine(value, scale, bias, index % channels));
            }
        }
        ImageAxisOrder::Chw => {
            for batch_index in 0..batch {
                let batch_start = batch_index * channels * spatial_size;
                for channel in 0..channels {
                    let start = batch_start + channel * spatial_size;
                    let end = start + spatial_size;
                    output.extend(
                        values[start..end]
                            .iter()
                            .map(|&value| apply_affine(value, scale, bias, channel)),
                    );
                }
            }
        }
    }
    output
}

fn normalize_contiguous_nhwc_to_nchw(
    values: &[u8],
    dims: &[usize; 4],
    scale: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let [batch, height, width, channels] = *dims;
    let spatial_size = height * width;
    let image_size = spatial_size * channels;
    let mut output = vec![0.0; values.len()];
    for batch_index in 0..batch {
        let source = &values[batch_index * image_size..(batch_index + 1) * image_size];
        for channel in 0..channels {
            let output_start = (batch_index * channels + channel) * spatial_size;
            for spatial_index in 0..spatial_size {
                output[output_start + spatial_index] = apply_affine(
                    source[spatial_index * channels + channel],
                    scale,
                    bias,
                    channel,
                );
            }
        }
    }
    output
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

    let channel_count = match axis_order {
        ImageAxisOrder::Hwc => input.dims()[3],
        ImageAxisOrder::Chw => input.dims()[1],
    };
    if mean.len() != 1 && mean.len() != channel_count {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channel_count,
            mean.len()
        )));
    }

    let (scale, bias) = affine_params(mean, std, channel_count);
    let dims = input.dims().to_vec();
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
            let output = normalize_contiguous_batch(values, &dims, axis_order, &scale, &bias);
            return Tensor::from_vec(output, dims, &device);
        }

        let [dim0, dim1, dim2, dim3] = dims.as_slice() else {
            return Err(rivet_core::Error::InvalidRank {
                expected: 4,
                actual: dims.len(),
            });
        };
        let stride = layout.stride();
        let mut output = Vec::with_capacity(elem_count);
        for index0 in 0..*dim0 {
            let offset0 = index0
                .checked_mul(stride[0])
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            for index1 in 0..*dim1 {
                let offset1 = offset0
                    .checked_add(
                        index1
                            .checked_mul(stride[1])
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                    )
                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                for index2 in 0..*dim2 {
                    let offset2 = offset1
                        .checked_add(
                            index2
                                .checked_mul(stride[2])
                                .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                        )
                        .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                    for index3 in 0..*dim3 {
                        let offset = offset2
                            .checked_add(
                                index3
                                    .checked_mul(stride[3])
                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                            )
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        let physical_index = layout
                            .start_offset()
                            .checked_add(offset)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        let value = *values
                            .get(physical_index)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        let channel = match axis_order {
                            ImageAxisOrder::Hwc => index3,
                            ImageAxisOrder::Chw => index1,
                        };
                        output.push(apply_affine(value, &scale, &bias, channel));
                    }
                }
            }
        }

        Tensor::from_vec(output, dims, &device)
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

    let [batch, height, width, channels] = input.dims() else {
        return Err(invalid_shape(format!(
            "normalize_u8_batch_to_nchw_f32 requires NHWC input, got shape {:?}",
            input.dims()
        )));
    };
    if mean.len() != 1 && mean.len() != *channels {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channels,
            mean.len()
        )));
    }

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
            let output = normalize_contiguous_nhwc_to_nchw(values, &source_dims, &scale, &bias);
            return Tensor::from_vec(output, dims, &device);
        }

        let stride = layout.stride();
        let mut output = vec![0.0; elem_count];
        for batch_index in 0..*batch {
            for height_index in 0..*height {
                for width_index in 0..*width {
                    let pixel_offset = batch_index
                        .checked_mul(stride[0])
                        .and_then(|offset| offset.checked_add(height_index.checked_mul(stride[1])?))
                        .and_then(|offset| offset.checked_add(width_index.checked_mul(stride[2])?))
                        .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                    for channel in 0..*channels {
                        let offset = pixel_offset
                            .checked_add(
                                channel
                                    .checked_mul(stride[3])
                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                            )
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        let physical_index = layout
                            .start_offset()
                            .checked_add(offset)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        let value = *values
                            .get(physical_index)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        let output_index =
                            ((batch_index * *channels + channel) * *height + height_index) * *width
                                + width_index;
                        output[output_index] = apply_affine(value, &scale, &bias, channel);
                    }
                }
            }
        }

        Tensor::from_vec(output, dims, &device)
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

    let channel_count = match axis_order {
        ImageAxisOrder::Hwc => input.dims()[2],
        ImageAxisOrder::Chw => input.dims()[0],
    };
    if mean.len() != 1 && mean.len() != channel_count {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channel_count,
            mean.len()
        )));
    }

    let spatial_size = match axis_order {
        ImageAxisOrder::Hwc => 1,
        ImageAxisOrder::Chw => input.dims()[1]
            .checked_mul(input.dims()[2])
            .ok_or_else(|| invalid_shape("normalize image dimensions overflow"))?,
    };
    let (scale, bias) = affine_params(mean, std, channel_count);
    let dims = input.dims().to_vec();
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
            let output = normalize_contiguous_image(values, &dims, axis_order, &scale, &bias);
            return Tensor::from_vec(output, dims, &device);
        }

        let mut output = Vec::with_capacity(elem_count);
        for (logical_index, physical_index) in layout.strided_index().enumerate() {
            let value = *values
                .get(physical_index)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            let channel = match axis_order {
                ImageAxisOrder::Hwc => logical_index % channel_count,
                ImageAxisOrder::Chw => logical_index / spatial_size,
            };
            output.push(apply_affine(value, &scale, &bias, channel));
        }

        Tensor::from_vec(output, dims, &device)
    })?)
}

#[cfg(test)]
mod tests {
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
}
