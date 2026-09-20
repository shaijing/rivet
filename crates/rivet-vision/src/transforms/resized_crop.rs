use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::pipeline::op::SampleContext;
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::InterpolationMode;
use crate::transforms::resize::ResizeConfig;
use rivet_core::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomResizedCropConfig {
    pub width: u32,
    pub height: u32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    pub interpolation: InterpolationMode,
}

impl RandomResizedCropConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            scale_min: 0.08,
            scale_max: 1.0,
            ratio_min: 0.75,
            ratio_max: 1.3333334,
            interpolation: InterpolationMode::Bilinear,
        }
    }

    pub const fn with_params(
        mut self,
        scale_min: f32,
        scale_max: f32,
        ratio_min: f32,
        ratio_max: f32,
    ) -> Self {
        self.scale_min = scale_min;
        self.scale_max = scale_max;
        self.ratio_min = ratio_min;
        self.ratio_max = ratio_max;
        self
    }

    pub const fn with_interpolation(mut self, interpolation: InterpolationMode) -> Self {
        self.interpolation = interpolation;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(invalid_argument(
                "random_resized_crop width and height must be greater than 0",
            ));
        }
        if !self.scale_min.is_finite()
            || !self.scale_max.is_finite()
            || self.scale_min <= 0.0
            || self.scale_min > self.scale_max
            || self.scale_max > 1.0
        {
            return Err(invalid_argument(
                "random_resized_crop scale must satisfy 0 < min <= max <= 1",
            ));
        }
        if !self.ratio_min.is_finite()
            || !self.ratio_max.is_finite()
            || self.ratio_min <= 0.0
            || self.ratio_min > self.ratio_max
        {
            return Err(invalid_argument(
                "random_resized_crop ratio must satisfy 0 < min <= max",
            ));
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        image_width: u32,
        image_height: u32,
        ctx: &mut SampleContext,
    ) -> CropRegion {
        if image_width == 0 || image_height == 0 {
            return CropRegion {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            };
        }
        let area = f64::from(image_width) * f64::from(image_height);
        let log_ratio_min = f64::from(self.ratio_min).ln();
        let log_ratio_max = f64::from(self.ratio_max).ln();
        for _ in 0..10 {
            let scale = uniform(
                ctx.next_rng_u64(),
                f64::from(self.scale_min),
                f64::from(self.scale_max),
            );
            let ratio = (uniform(ctx.next_rng_u64(), log_ratio_min, log_ratio_max)).exp();
            let target_area = area * scale;
            let crop_width = (target_area * ratio).sqrt().round() as u32;
            let crop_height = (target_area / ratio).sqrt().round() as u32;
            if crop_width > 0
                && crop_height > 0
                && crop_width <= image_width
                && crop_height <= image_height
            {
                let x = if image_width == crop_width {
                    0
                } else {
                    (ctx.next_rng_u64() % u64::from(image_width - crop_width + 1)) as u32
                };
                let y = if image_height == crop_height {
                    0
                } else {
                    (ctx.next_rng_u64() % u64::from(image_height - crop_height + 1)) as u32
                };
                return CropRegion {
                    x,
                    y,
                    width: crop_width,
                    height: crop_height,
                };
            }
        }

        let image_ratio = f64::from(image_width) / f64::from(image_height.max(1));
        let (crop_width, crop_height) = if image_ratio < f64::from(self.ratio_min) {
            (
                image_width,
                (f64::from(image_width) / f64::from(self.ratio_min)).round() as u32,
            )
        } else if image_ratio > f64::from(self.ratio_max) {
            (
                (f64::from(image_height) * f64::from(self.ratio_max)).round() as u32,
                image_height,
            )
        } else {
            (image_width, image_height)
        };
        let crop_width = crop_width.clamp(1, image_width.max(1));
        let crop_height = crop_height.clamp(1, image_height.max(1));
        CropRegion {
            x: (image_width.saturating_sub(crop_width)) / 2,
            y: (image_height.saturating_sub(crop_height)) / 2,
            width: crop_width,
            height: crop_height,
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        if axis_order != ImageAxisOrder::Hwc {
            return Err(invalid_argument(
                "random_resized_crop currently requires uint8 HWC input",
            ));
        }
        let sample = sample.into_decoded()?;
        let dims = sample.image.dims();
        if sample.image.dtype() != DType::U8 || dims.len() != 3 || dims[2] != 3 {
            return Err(invalid_argument(format!(
                "random_resized_crop requires uint8 HWC RGB input, got dtype {:?} shape {:?}",
                sample.image.dtype(),
                dims
            )));
        }
        let image_width = u32::try_from(dims[1])
            .map_err(|_| invalid_shape("random_resized_crop image width is too large"))?;
        let image_height = u32::try_from(dims[0])
            .map_err(|_| invalid_shape("random_resized_crop image height is too large"))?;
        if image_width == 0 || image_height == 0 {
            return Err(invalid_shape(
                "random_resized_crop does not support empty images",
            ));
        }
        let region = self.resolve(image_width, image_height, ctx);
        let image = sample
            .image
            .narrow(0, region.y as usize, region.height as usize)?
            .narrow(1, region.x as usize, region.width as usize)?;
        ResizeConfig::with_interpolation(self.width, self.height, self.interpolation).apply(
            ImageSample::Decoded(DecodedSample {
                image,
                label: sample.label,
            }),
        )
    }
}

fn uniform(bits: u64, min: f64, max: f64) -> f64 {
    let unit = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    min + (max - min) * unit
}

#[cfg(test)]
mod tests {
    use super::RandomResizedCropConfig;
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    fn sample() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec((0..48).collect::<Vec<u8>>(), [4, 4, 3], &Device::Cpu).unwrap(),
            label: 1,
        })
    }

    #[test]
    fn random_resized_crop_is_deterministic_and_has_requested_shape() {
        let config = RandomResizedCropConfig::new(2, 3);
        let mut left = SampleContext::new(5);
        left.global_seed = 42;
        let mut right = SampleContext::new(5);
        right.global_seed = 42;
        let left = config
            .apply(sample(), &mut left, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        let right = config
            .apply(sample(), &mut right, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(left.image.dims(), [3, 2, 3]);
        assert_eq!(left.image.dtype(), DType::U8);
        assert_eq!(
            left.image.to_vec::<u8>().unwrap(),
            right.image.to_vec::<u8>().unwrap()
        );
    }

    #[test]
    fn random_resized_crop_fallback_is_centered() {
        let config = RandomResizedCropConfig::new(2, 2).with_params(1.0, 1.0, 0.1, 0.5);
        let mut ctx = SampleContext::new(0);
        let region = config.resolve(8, 4, &mut ctx);
        assert_eq!(
            region,
            super::CropRegion {
                x: 3,
                y: 0,
                width: 2,
                height: 4
            }
        );
    }
}
