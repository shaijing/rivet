use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::blur;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlurConfig {
    pub sigma: f32,
}

impl GaussianBlurConfig {
    pub const fn new(sigma: f32) -> Self {
        Self { sigma }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.sigma.is_finite() && self.sigma >= 0.0 {
            Ok(())
        } else {
            Err(invalid_argument(
                "gaussian_blur sigma must be finite and non-negative",
            ))
        }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "gaussian_blur")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            blur(&image, self.sigma),
            label,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianBlurConfig;
    use crate::sample::image::{DecodedSample, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn gaussian_blur_materializes_and_preserves_shape_dtype() {
        let input =
            Tensor::from_vec(vec![255u8, 0, 0, 0, 0, 255], [1, 2, 3], &Device::Cpu).unwrap();
        let output = GaussianBlurConfig::new(1.0)
            .apply(ImageSample::Decoded(DecodedSample {
                image: input.clone(),
                label: 2,
            }))
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert!(!output.image.same_storage(&input));
    }

    #[test]
    fn gaussian_blur_rejects_invalid_sigma() {
        assert!(GaussianBlurConfig::new(-1.0).validate().is_err());
        assert!(GaussianBlurConfig::new(f32::NAN).validate().is_err());
    }
}
