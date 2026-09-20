use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::{FilterType, resize};
use std::str::FromStr;

/// Interpolation policy owned by Rivet rather than the image backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterpolationMode {
    Nearest,
    /// Bilinear interpolation, implemented by the backend's triangle filter.
    #[default]
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl InterpolationMode {
    fn filter_type(self) -> FilterType {
        match self {
            Self::Nearest => FilterType::Nearest,
            Self::Bilinear => FilterType::Triangle,
            Self::Bicubic => FilterType::CatmullRom,
            Self::Lanczos3 => FilterType::Lanczos3,
        }
    }
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
        let (image, label) = into_rgb_image(sample, "resize")?;
        let resized = resize(
            &image,
            self.width,
            self.height,
            self.interpolation.filter_type(),
        );
        Ok(ImageSample::Decoded(from_rgb_image(resized, label)?))
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
