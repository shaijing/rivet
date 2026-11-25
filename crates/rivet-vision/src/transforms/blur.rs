use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::with_rgb_pixels;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlurConfig {
    pub sigma: f32,
    /// Standard deviations in torchvision's (horizontal, vertical) order.
    pub axis_sigma: Option<[f32; 2]>,
    pub kernel_size: Option<u32>,
    /// Kernel dimensions in torchvision's (width, height) order.
    pub kernel_dimensions: Option<[u32; 2]>,
}

impl GaussianBlurConfig {
    pub const fn new(sigma: f32) -> Self {
        Self {
            sigma,
            axis_sigma: None,
            kernel_size: None,
            kernel_dimensions: None,
        }
    }

    pub const fn with_kernel_size(mut self, kernel_size: u32) -> Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    pub const fn with_axis_sigmas(mut self, sigma: [f32; 2]) -> Self {
        self.axis_sigma = Some(sigma);
        self
    }

    pub const fn with_kernel_dimensions(mut self, dimensions: [u32; 2]) -> Self {
        self.kernel_dimensions = Some(dimensions);
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.kernel_size.is_some_and(|k| k == 0 || k % 2 == 0)
            || self
                .kernel_dimensions
                .is_some_and(|ks| ks.iter().any(|k| *k == 0 || *k % 2 == 0))
        {
            return Err(invalid_argument(
                "gaussian_blur kernel size must be positive and odd",
            ));
        }
        let sigmas = self.axis_sigma.unwrap_or([self.sigma; 2]);
        if sigmas.iter().all(|s| s.is_finite() && *s >= 0.0) {
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
        let image = with_rgb_pixels(&sample.image, "gaussian_blur", |pixels, width, height| {
            let sigmas = self.axis_sigma.unwrap_or([self.sigma; 2]);
            if sigmas == [0.0, 0.0] {
                return Ok(rivet_core::Tensor::from_exact_iter(
                    pixels.iter().copied(),
                    [height, width, 3],
                    sample.image.device(),
                )?);
            }
            let default_kernel = |sigma: f32| {
                ((3.0 * sigma).ceil() as usize)
                    .checked_mul(2)
                    .and_then(|size| size.checked_add(1))
                    .ok_or_else(|| invalid_argument("gaussian_blur kernel size overflows usize"))
            };
            let dimensions = self.kernel_dimensions.unwrap_or_else(|| {
                let size = self.kernel_size.unwrap_or(0);
                [size, size]
            });
            let kernel_x = if dimensions[0] > 0 {
                dimensions[0] as usize
            } else {
                default_kernel(sigmas[0])?
            };
            let kernel_y = if dimensions[1] > 0 {
                dimensions[1] as usize
            } else {
                default_kernel(sigmas[1])?
            };
            let radius_x = kernel_x / 2;
            let radius_y = kernel_y / 2;
            if radius_x >= width || radius_y >= height {
                return Err(invalid_argument(
                    "gaussian_blur reflection padding must be smaller than image dimensions",
                ));
            }
            let weights = |kernel: usize, sigma: f32| -> Vec<f32> {
                if sigma == 0.0 {
                    let mut w = vec![0.0; kernel];
                    w[kernel / 2] = 1.0;
                    return w;
                }
                let weights: Vec<f32> = (0..kernel)
                    .map(|i| {
                        let x = (i as f32 - (kernel / 2) as f32) / sigma;
                        (-0.5 * x * x).exp()
                    })
                    .collect();
                let total: f32 = weights.iter().sum();
                weights.into_iter().map(|w| w / total).collect()
            };
            let weights_x = weights(kernel_x, sigmas[0]);
            let weights_y = weights(kernel_y, sigmas[1]);
            let reflect = |i: isize, len: usize| {
                if i < 0 {
                    (-i) as usize
                } else if i >= len as isize {
                    2 * len - i as usize - 2
                } else {
                    i as usize
                }
            };
            let image = rivet_core::Tensor::from_exact_writer::<u8, _, _>(
                [height, width, 3],
                &rivet_core::Device::Cpu,
                |out| {
                    for y in 0..height {
                        for x in 0..width {
                            let mut sums = [0.0f32; 3];
                            for (ky, wy) in weights_y.iter().enumerate() {
                                let sy =
                                    reflect(y as isize + ky as isize - radius_y as isize, height);
                                for (kx, wx) in weights_x.iter().enumerate() {
                                    let sx = reflect(
                                        x as isize + kx as isize - radius_x as isize,
                                        width,
                                    );
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
            Ok(image)
        })?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
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

pub(super) fn motion_blur(
    input: &rivet_core::Tensor,
    length: u32,
    angle: f64,
) -> RivetResult<rivet_core::Tensor> {
    with_rgb_pixels(input, "MotionBlur", |pixels, width, height| {
        let (dy, dx) = angle.to_radians().sin_cos();
        let radius = i64::from(length / 2);
        let taps = (-radius..=radius)
            .map(|i| {
                let i = i as f64;
                ((i * dy).round() as i64, (i * dx).round() as i64)
            })
            .collect::<Vec<_>>();
        let reflect = |index: i64, len: usize| {
            let len = len as i64;
            if (0..len).contains(&index) {
                return index as usize;
            }
            if len <= 1 {
                return 0;
            }
            let period = 2 * (len - 1);
            let index = index.rem_euclid(period);
            index.min(period - index) as usize
        };
        Ok(rivet_core::Tensor::from_exact_writer::<u8, _, _>(
            [height, width, 3],
            input.device(),
            |out| {
                for y in 0..height {
                    for x in 0..width {
                        let mut sums = [0u64; 3];
                        for &(dy, dx) in &taps {
                            let sy = reflect(y as i64 + dy, height);
                            let sx = reflect(x as i64 + dx, width);
                            let offset = (sy * width + sx) * 3;
                            sums[0] += u64::from(pixels[offset]);
                            sums[1] += u64::from(pixels[offset + 1]);
                            sums[2] += u64::from(pixels[offset + 2]);
                        }
                        let count = u64::from(length);
                        for sum in sums {
                            out.write_next(((sum + count / 2) / count) as u8)?;
                        }
                    }
                }
                Ok(())
            },
        )?)
    })
}

pub(super) fn unsharp_mask(
    sample: crate::sample::image::DecodedSample,
    sigma: f32,
    amount: f32,
    threshold: u8,
) -> RivetResult<rivet_core::Tensor> {
    let input = &sample.image;
    let [height, width, _] = *input.dims() else {
        return Err(invalid_argument("UnsharpMask requires a rank-3 image"));
    };
    if amount == 0.0 {
        return Ok(input.clone());
    }
    // Bound the radius for the existing reflect-padding Gaussian kernel.
    let radius = |len: usize| {
        ((3.0 * sigma).ceil() as u32).min((len.saturating_sub(1) as u32).min(u32::MAX / 2))
    };
    let blurred = GaussianBlurConfig::new(sigma)
        .with_kernel_dimensions([2 * radius(width) + 1, 2 * radius(height) + 1])
        .apply(ImageSample::Decoded(sample.clone()))?
        .into_decoded()?;
    with_rgb_pixels(input, "UnsharpMask", |pixels, _, _| {
        with_rgb_pixels(&blurred.image, "UnsharpMask", |smooth, _, _| {
            Ok(rivet_core::Tensor::from_exact_iter(
                pixels.iter().zip(smooth).map(|(&pixel, &blurred)| {
                    let detail = i16::from(pixel) - i16::from(blurred);
                    if detail.unsigned_abs() > u16::from(threshold) {
                        (f32::from(pixel) + amount * f32::from(detail))
                            .round()
                            .clamp(0.0, 255.0) as u8
                    } else {
                        pixel
                    }
                }),
                input.dims(),
                input.device(),
            )?)
        })
    })
}
