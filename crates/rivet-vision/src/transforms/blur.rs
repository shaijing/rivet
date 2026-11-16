use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::{from_rgb_image, into_rgb_image};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlurConfig {
    pub sigma: f32,
    pub kernel_size: Option<u32>,
}

impl GaussianBlurConfig {
    pub const fn new(sigma: f32) -> Self {
        Self {
            sigma,
            kernel_size: None,
        }
    }

    pub const fn with_kernel_size(mut self, kernel_size: u32) -> Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.kernel_size.is_some_and(|k| k == 0 || k % 2 == 0) {
            return Err(invalid_argument(
                "gaussian_blur kernel size must be positive and odd",
            ));
        }
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
        let (width, height) = (image.width() as usize, image.height() as usize);
        if self.sigma == 0.0 {
            return Ok(ImageSample::Decoded(from_rgb_image(image, label)?));
        }
        let kernel = match self.kernel_size {
            Some(size) => size as usize,
            None => ((3.0 * self.sigma).ceil() as usize)
                .checked_mul(2)
                .and_then(|size| size.checked_add(1))
                .ok_or_else(|| invalid_argument("gaussian_blur kernel size overflows usize"))?,
        };
        let radius = kernel / 2;
        if radius >= width || radius >= height {
            return Err(invalid_argument(
                "gaussian_blur reflection padding must be smaller than image dimensions",
            ));
        }
        let weights: Vec<f32> = (0..kernel)
            .map(|i| {
                let x = (i as f32 - radius as f32) / self.sigma;
                (-0.5 * x * x).exp()
            })
            .collect();
        let total: f32 = weights.iter().sum();
        let weights: Vec<f32> = weights.into_iter().map(|w| w / total).collect();
        let reflect = |i: isize, len: usize| {
            if i < 0 {
                (-i) as usize
            } else if i >= len as isize {
                2 * len - i as usize - 2
            } else {
                i as usize
            }
        };
        let pixels = image.as_raw();
        let image = rivet_core::Tensor::from_exact_writer::<u8, _, _>(
            [height, width, 3],
            &rivet_core::Device::Cpu,
            |out| {
                for y in 0..height {
                    for x in 0..width {
                        let mut sums = [0.0f32; 3];
                        for (ky, wy) in weights.iter().enumerate() {
                            let sy = reflect(y as isize + ky as isize - radius as isize, height);
                            for (kx, wx) in weights.iter().enumerate() {
                                let sx = reflect(x as isize + kx as isize - radius as isize, width);
                                let offset = (sy * width + sx) * 3;
                                let weight = wy * wx;
                                sums[0] += f32::from(pixels[offset]) * weight;
                                sums[1] += f32::from(pixels[offset + 1]) * weight;
                                sums[2] += f32::from(pixels[offset + 2]) * weight;
                            }
                        }
                        out.write_next(sums[0].round_ties_even().clamp(0.0, 255.0) as u8)?;
                        out.write_next(sums[1].round_ties_even().clamp(0.0, 255.0) as u8)?;
                        out.write_next(sums[2].round_ties_even().clamp(0.0, 255.0) as u8)?;
                    }
                }
                Ok(())
            },
        )?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianBlurConfig;
    use crate::sample::image::{DecodedSample, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn gaussian_blur_materializes_and_preserves_shape_dtype() {
        let input = Tensor::from_vec(vec![127u8; 7 * 7 * 3], [7, 7, 3], &Device::Cpu).unwrap();
        let output = GaussianBlurConfig::new(1.0)
            .apply(ImageSample::Decoded(DecodedSample {
                image: input.clone(),
                label: 2,
            }))
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [7, 7, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert!(!output.image.same_storage(&input));
    }

    #[test]
    fn gaussian_blur_rejects_invalid_sigma() {
        assert!(GaussianBlurConfig::new(-1.0).validate().is_err());
        assert!(GaussianBlurConfig::new(f32::NAN).validate().is_err());
    }
}
