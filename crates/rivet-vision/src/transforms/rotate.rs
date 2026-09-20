use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::{rotate90, rotate180, rotate270};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationAngle {
    Deg90,
    Deg180,
    Deg270,
}

impl TryFrom<i32> for RotationAngle {
    type Error = crate::errors::VisionError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            90 => Ok(Self::Deg90),
            180 => Ok(Self::Deg180),
            270 => Ok(Self::Deg270),
            _ => Err(invalid_argument(
                "rotation angle must be one of 90, 180, or 270 degrees",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotateConfig {
    pub angle: RotationAngle,
}

impl RotateConfig {
    pub const fn new(angle: RotationAngle) -> Self {
        Self { angle }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "rotate")?;
        let rotated = match self.angle {
            RotationAngle::Deg90 => rotate90(&image),
            RotationAngle::Deg180 => rotate180(&image),
            RotationAngle::Deg270 => rotate270(&image),
        };
        Ok(ImageSample::Decoded(from_rgb_image(rotated, label)?))
    }
}

#[cfg(test)]
mod tests {
    use super::{RotateConfig, RotationAngle};
    use crate::sample::image::{DecodedSample, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn fixed_rotation_changes_spatial_shape_and_preserves_dtype() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], [2, 1, 3], &Device::Cpu).unwrap();
        let output = RotateConfig::new(RotationAngle::Deg90)
            .apply(ImageSample::Decoded(DecodedSample {
                image: input,
                label: 1,
            }))
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn rotation_accepts_only_fixed_angles() {
        assert!(RotationAngle::try_from(90).is_ok());
        assert!(RotationAngle::try_from(45).is_err());
    }
}
