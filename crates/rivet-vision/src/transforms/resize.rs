use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::with_rgb_pixels;
use std::str::FromStr;

/// Interpolation policy owned by Rivet rather than the image backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterpolationMode {
    Nearest,
    /// Pillow-compatible triangle filter with uint8 pass rounding.
    #[default]
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl FromStr for InterpolationMode {
    type Err = crate::errors::VisionError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "nearest" => Ok(Self::Nearest),
            "bilinear" | "linear" => Ok(Self::Bilinear),
            "bicubic" | "cubic" => Ok(Self::Bicubic),
            "lanczos3" | "lanczos" => Ok(Self::Lanczos3),
            _ => Err(invalid_argument(format!(
                "unknown interpolation mode {value:?}; expected nearest, bilinear, bicubic, or lanczos3"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizeConfig {
    pub width: u32,
    pub height: u32,
    pub interpolation: InterpolationMode,
}

impl ResizeConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            interpolation: InterpolationMode::Bilinear,
        }
    }

    pub const fn with_interpolation(
        width: u32,
        height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        Self {
            width,
            height,
            interpolation,
        }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let image = with_rgb_pixels(&sample.image, "resize", |pixels, width, height| {
            resize_rgb(
                pixels,
                width,
                height,
                self.width,
                self.height,
                self.interpolation,
            )
        })?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
        }))
    }
}

// Pillow Resample.c: separable RGB/u8 resampling with signed 22-bit
// coefficients. Each pass clips and rounds to u8 before the next pass.
const PRECISION: i64 = 1 << 22;

fn filter(mode: InterpolationMode, x: f64) -> f64 {
    let x = x.abs();
    match mode {
        InterpolationMode::Bilinear => (1.0 - x).max(0.0),
        InterpolationMode::Bicubic if x < 1.0 => (1.5 * x - 2.5) * x * x + 1.0,
        InterpolationMode::Bicubic if x < 2.0 => (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5,
        InterpolationMode::Lanczos3 if x < 3.0 => {
            let sinc = |x: f64| {
                if x == 0.0 {
                    1.0
                } else {
                    let x = x * std::f64::consts::PI;
                    x.sin() / x
                }
            };
            sinc(x) * sinc(x / 3.0)
        }
        _ => 0.0,
    }
}

struct Coefficients {
    start: usize,
    weights: Vec<i32>,
}

fn coefficients(input: usize, output: usize, mode: InterpolationMode) -> Vec<Coefficients> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter_scale
        * match mode {
            InterpolationMode::Bilinear => 1.0,
            InterpolationMode::Bicubic => 2.0,
            InterpolationMode::Lanczos3 => 3.0,
            InterpolationMode::Nearest => unreachable!(),
        };
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (start..end)
                .map(|x| filter(mode, (x as f64 - center + 0.5) / filter_scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            Coefficients {
                start,
                weights: weights
                    .into_iter()
                    .map(|w| {
                        let w = if sum != 0.0 { w / sum } else { w };
                        (w * PRECISION as f64).round() as i32
                    })
                    .collect(),
            }
        })
        .collect()
}

pub(crate) fn resize_rgb(
    source: &[u8],
    iw: usize,
    ih: usize,
    width: u32,
    height: u32,
    mode: InterpolationMode,
) -> RivetResult<rivet_core::Tensor> {
    use rivet_core::{Device, Tensor};
    if width == 0 || height == 0 {
        return Err(invalid_argument(
            "resize width and height must be greater than 0",
        ));
    }
    let (ow, oh) = (width as usize, height as usize);
    if iw == 0 || ih == 0 {
        return Err(invalid_argument("resize requires a non-empty input image"));
    }
    if mode == InterpolationMode::Nearest {
        // Pillow's affine nearest path increments coordinates instead of
        // multiplying for each pixel (observable at exact half-pixel ties).
        let scale_x = iw as f64 / ow as f64;
        let mut position = scale_x * 0.5;
        let xs: Vec<usize> = (0..ow)
            .map(|_| {
                let x = (position as usize).min(iw - 1);
                position += scale_x;
                x
            })
            .collect();
        return Ok(Tensor::from_exact_writer::<u8, _, _>(
            [oh, ow, 3],
            &Device::Cpu,
            |out| {
                let scale_y = ih as f64 / oh as f64;
                let mut position = scale_y * 0.5;
                for _ in 0..oh {
                    let y = (position as usize).min(ih - 1);
                    position += scale_y;
                    for &x in &xs {
                        let offset = (y * iw + x) * 3;
                        out.write_next(source[offset])?;
                        out.write_next(source[offset + 1])?;
                        out.write_next(source[offset + 2])?;
                    }
                }
                Ok(())
            },
        )?);
    }
    let clip = |value: i64| ((value + (PRECISION >> 1)) >> 22).clamp(0, 255) as u8;
    let horizontal = if iw != ow {
        let coeffs = coefficients(iw, ow, mode);
        Some(Tensor::from_exact_writer::<u8, _, _>(
            [ih, ow, 3],
            &Device::Cpu,
            |out| {
                for y in 0..ih {
                    for coeff in &coeffs {
                        let mut sums = [0i64; 3];
                        let start = (y * iw + coeff.start) * 3;
                        for (pixel, &weight) in source[start..start + coeff.weights.len() * 3]
                            .chunks_exact(3)
                            .zip(&coeff.weights)
                        {
                            sums[0] += i64::from(pixel[0]) * i64::from(weight);
                            sums[1] += i64::from(pixel[1]) * i64::from(weight);
                            sums[2] += i64::from(pixel[2]) * i64::from(weight);
                        }
                        out.write_next(clip(sums[0]))?;
                        out.write_next(clip(sums[1]))?;
                        out.write_next(clip(sums[2]))?;
                    }
                }
                Ok(())
            },
        )?)
    } else {
        None
    };
    if ih == oh {
        return match horizontal {
            Some(tensor) => Ok(tensor),
            None => Ok(Tensor::from_exact_iter(
                source.iter().copied(),
                [oh, ow, 3],
                &Device::Cpu,
            )?),
        };
    }
    let coeffs = coefficients(ih, oh, mode);
    // The horizontal result is the required u8 pass boundary, not a staging
    // Vec. Vertical output goes straight into the final aligned tensor.
    let vertical = |pixels: &[u8]| {
        Tensor::from_exact_writer::<u8, _, _>([oh, ow, 3], &Device::Cpu, |out| {
            for coeff in &coeffs {
                for x in 0..ow {
                    let mut sums = [0i64; 3];
                    for (row, &weight) in coeff.weights.iter().enumerate() {
                        let offset = ((coeff.start + row) * ow + x) * 3;
                        sums[0] += i64::from(pixels[offset]) * i64::from(weight);
                        sums[1] += i64::from(pixels[offset + 1]) * i64::from(weight);
                        sums[2] += i64::from(pixels[offset + 2]) * i64::from(weight);
                    }
                    out.write_next(clip(sums[0]))?;
                    out.write_next(clip(sums[1]))?;
                    out.write_next(clip(sums[2]))?;
                }
            }
            Ok(())
        })
    };
    match horizontal {
        Some(tensor) => Ok(tensor.with_cpu_storage(|storage, _| {
            let rivet_core::CpuStorageRef::U8(pixels) = storage else {
                unreachable!()
            };
            vertical(pixels)
        })?),
        None => Ok(vertical(source)?),
    }
}

#[cfg(test)]
mod tests {
    use super::{InterpolationMode, ResizeConfig};
    use crate::sample::image::{DecodedSample, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn resize_config_defaults_to_bilinear() {
        assert_eq!(
            ResizeConfig::new(32, 24).interpolation,
            InterpolationMode::Bilinear
        );
    }

    #[test]
    fn interpolation_mode_parsing_is_backend_independent() {
        assert_eq!(
            "nearest".parse::<InterpolationMode>().unwrap(),
            InterpolationMode::Nearest
        );
        assert_eq!(
            "linear".parse::<InterpolationMode>().unwrap(),
            InterpolationMode::Bilinear
        );
        assert_eq!(
            "CUBIC".parse::<InterpolationMode>().unwrap(),
            InterpolationMode::Bicubic
        );
        assert_eq!(
            "lanczos".parse::<InterpolationMode>().unwrap(),
            InterpolationMode::Lanczos3
        );
        assert!("box".parse::<InterpolationMode>().is_err());
    }

    #[test]
    fn resize_materializes_non_contiguous_hwc_input() {
        let base = Tensor::from_vec((0..18).collect::<Vec<u8>>(), [2, 3, 3], &Device::Cpu).unwrap();
        let input = base.permute(&[1, 0, 2]).unwrap();
        assert!(!input.is_contiguous());
        let output = ResizeConfig::with_interpolation(4, 2, InterpolationMode::Nearest)
            .apply(ImageSample::Decoded(DecodedSample {
                image: input.clone(),
                label: 0,
            }))
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [2, 4, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert!(!output.image.same_storage(&input));
    }

    #[test]
    fn resize_executes_each_interpolation_mode_at_the_backend_boundary() {
        let modes = [
            InterpolationMode::Nearest,
            InterpolationMode::Bilinear,
            InterpolationMode::Bicubic,
            InterpolationMode::Lanczos3,
        ];
        for mode in modes {
            let input = Tensor::from_vec(
                vec![
                    0u8, 1, 2, // top-left
                    10, 11, 12, // top-right
                    20, 21, 22, // bottom-left
                    30, 31, 32, // bottom-right
                ],
                [2, 2, 3],
                &Device::Cpu,
            )
            .unwrap();
            let output = ResizeConfig::with_interpolation(4, 4, mode)
                .apply(ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 7,
                }))
                .unwrap()
                .into_decoded()
                .unwrap();

            assert_eq!(output.image.dims(), [4, 4, 3]);
            assert_eq!(output.image.dtype(), DType::U8);
            assert_eq!(output.label, 7);
            assert!(!output.image.same_storage(&input));
            if mode == InterpolationMode::Nearest {
                let values = output.image.to_vec::<u8>().unwrap();
                assert_eq!(&values[..3], &[0, 1, 2]);
                assert_eq!(&values[45..48], &[30, 31, 32]);
            }
        }

        let input = Tensor::zeros([2, 2, 3], DType::F32, &Device::Cpu).unwrap();
        let err = ResizeConfig::new(4, 4)
            .apply(ImageSample::Decoded(DecodedSample {
                image: input,
                label: 0,
            }))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires uint8 HWC RGB"), "got: {err}");
    }
}
