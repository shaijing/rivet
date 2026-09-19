use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
use rivet_core::{CpuStorageRef, DType, Device, Tensor};

#[derive(Clone)]
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

    pub fn apply(&self, sample: ImageSample, layout: ImageLayout) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let dims = sample.image.dims();
        if dims.len() != 3 {
            return Err(invalid_shape(format!(
                "normalize requires a rank-3 image, got shape {:?}",
                dims
            )));
        }
        let channel_count = match layout {
            ImageLayout::Hwc => dims[2],
            ImageLayout::Chw => dims[0],
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
            ImageLayout::Hwc => [1, 1, self.mean.len()],
            ImageLayout::Chw => [self.mean.len(), 1, 1],
        };
        let mean = Tensor::from_vec(self.mean.clone(), stats_shape, &Device::Cpu)?;
        let std = Tensor::from_vec(self.std.clone(), stats_shape, &Device::Cpu)?;
        let image = values.broadcast_sub(&mean)?.broadcast_div(&std)?;

        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: sample.label,
        }))
    }

    /// Apply normalization to a stacked rank-4 image batch. This is the
    /// initial batch-stage implementation; the one-allocation fused kernel
    /// is intentionally left for the batch-normalize phase.
    pub fn apply_batch(&self, input: Tensor, layout: ImageLayout) -> RivetResult<Tensor> {
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
            ImageLayout::Hwc => input.dims()[3],
            ImageLayout::Chw => input.dims()[1],
        };
        if self.mean.len() != 1 && self.mean.len() != channel_count {
            return Err(invalid_argument(format!(
                "normalize mean/std length must be 1 or channel count {}, got {}",
                channel_count,
                self.mean.len()
            )));
        }

        let values = match input.dtype() {
            DType::U8 => input.to_dtype(DType::F32)?.div_scalar(255.0f32)?,
            DType::F32 => input,
            dtype => {
                return Err(invalid_argument(format!(
                    "normalize supports uint8 or float32 input, got {:?}",
                    dtype
                )));
            }
        };

        let stats_shape = match layout {
            ImageLayout::Hwc => vec![1, 1, 1, self.mean.len()],
            ImageLayout::Chw => vec![1, self.mean.len(), 1, 1],
        };
        let mean = Tensor::from_vec(self.mean.clone(), stats_shape.clone(), &Device::Cpu)?;
        let std = Tensor::from_vec(self.std.clone(), stats_shape, &Device::Cpu)?;
        Ok(values.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }
}

/// Fused uint8 image normalization. The input is read in logical tensor
/// order and the output is allocated exactly once:
/// `((input as f32) / 255.0 - mean[channel]) / std[channel]`.
pub fn normalize_u8_to_f32(
    input: &Tensor,
    mean: &[f32],
    std: &[f32],
    axis_order: ImageLayout,
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
        ImageLayout::Hwc => input.dims()[2],
        ImageLayout::Chw => input.dims()[0],
    };
    if mean.len() != 1 && mean.len() != channel_count {
        return Err(invalid_argument(format!(
            "normalize mean/std length must be 1 or channel count {}, got {}",
            channel_count,
            mean.len()
        )));
    }

    let spatial_size = match axis_order {
        ImageLayout::Hwc => 1,
        ImageLayout::Chw => input.dims()[1]
            .checked_mul(input.dims()[2])
            .ok_or_else(|| invalid_shape("normalize image dimensions overflow"))?,
    };
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

        let mut output = Vec::with_capacity(elem_count);
        for (logical_index, physical_index) in layout.strided_index().enumerate() {
            let value = *values
                .get(physical_index)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            let channel = match axis_order {
                ImageLayout::Hwc => logical_index % channel_count,
                ImageLayout::Chw => logical_index / spatial_size,
            };
            let stats_index = if mean.len() == 1 { 0 } else { channel };
            output.push((value as f32 / 255.0 - mean[stats_index]) / std[stats_index]);
        }

        Tensor::from_vec(output, dims, &device)
    })?)
}

#[cfg(test)]
mod tests {
    use super::{NormalizeConfig, normalize_u8_to_f32};
    use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn normalize_config_uses_fused_u8_path() {
        let image = Tensor::from_vec(vec![0u8, 255, 128], [1, 1, 3], &Device::Cpu).unwrap();
        let sample = ImageSample::Decoded(DecodedSample { image, label: 0 });
        let out = NormalizeConfig::new(vec![0.5], vec![0.5])
            .apply(sample, ImageLayout::Hwc)
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
        let out = normalize_u8_to_f32(&chw, &[0.0, 0.5, 1.0], &[1.0, 0.5, 0.25], ImageLayout::Chw)
            .unwrap();

        assert_eq!(out.dims(), [3, 1, 2]);
        let values = out.to_vec::<f32>().unwrap();
        let expected = [0.0, 1.0, -0.49803922, -0.7490196, -1.9921569, -2.4941177];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn batch_normalize_supports_nhwc_and_chw() {
        let input =
            Tensor::from_vec(vec![0u8, 64, 128, 255, 32, 96], [1, 1, 2, 3], &Device::Cpu).unwrap();
        let config = NormalizeConfig::new(vec![0.0, 0.5, 1.0], vec![1.0, 0.5, 0.25]);

        let nhwc = config.apply_batch(input.clone(), ImageLayout::Hwc).unwrap();
        assert_eq!(nhwc.dims(), [1, 1, 2, 3]);
        let nhwc_values = nhwc.to_vec::<f32>().unwrap();
        let expected_nhwc = [0.0, -0.49803922, -1.9921569, 1.0, -0.7490196, -2.4941177];
        for (actual, expected) in nhwc_values.iter().zip(expected_nhwc) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }

        let nchw = config
            .apply_batch(input.permute(&[0, 3, 1, 2]).unwrap(), ImageLayout::Chw)
            .unwrap();
        assert_eq!(nchw.dims(), [1, 3, 1, 2]);
        let nchw_values = nchw.to_vec::<f32>().unwrap();
        let expected_nchw = [0.0, 1.0, -0.49803922, -0.7490196, -1.9921569, -2.4941177];
        for (actual, expected) in nchw_values.iter().zip(expected_nchw) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
}
