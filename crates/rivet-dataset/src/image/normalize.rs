use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
use rivet_core::{DType, Device, Tensor};

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
            sample.image.to_dtype(DType::F32)?.div_scalar(255.0f32)?
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
}

#[cfg(test)]
mod tests {
    use super::NormalizeConfig;
    use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn normalize_u8_to_f32() {
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
}
