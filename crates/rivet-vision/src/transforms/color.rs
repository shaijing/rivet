use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::{brighten, contrast, huerotate};
use rivet_core::{CpuStorageRef, DType, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrightnessConfig {
    pub value: i32,
}

impl BrightnessConfig {
    pub const fn new(value: i32) -> Self {
        Self { value }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "brightness")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            brighten(&image, self.value),
            label,
        )?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastConfig {
    pub value: f32,
}

impl ContrastConfig {
    pub const fn new(value: f32) -> Self {
        Self { value }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.value.is_finite() {
            Ok(())
        } else {
            Err(invalid_argument("contrast value must be finite"))
        }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "contrast")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            contrast(&image, self.value),
            label,
        )?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HueConfig {
    pub degrees: i32,
}

impl HueConfig {
    pub const fn new(degrees: i32) -> Self {
        Self { degrees }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "hue")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            huerotate(&image, self.degrees),
            label,
        )?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrayscaleConfig {
    /// Output channels. `1` produces Luma-like output; `3` replicates the
    /// luma value over RGB channels.
    pub num_output_channels: u8,
}

impl GrayscaleConfig {
    pub const fn new(num_output_channels: u8) -> Self {
        Self {
            num_output_channels,
        }
    }

    pub const fn one_channel() -> Self {
        Self::new(1)
    }

    pub fn validate(&self) -> RivetResult<()> {
        if matches!(self.num_output_channels, 1 | 3) {
            Ok(())
        } else {
            Err(invalid_argument(
                "grayscale num_output_channels must be 1 or 3",
            ))
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let input = &sample.image;
        if input.dtype() != DType::U8 || input.rank() != 3 {
            return Err(invalid_argument(format!(
                "grayscale requires a rank-3 uint8 image, got dtype {:?} shape {:?}",
                input.dtype(),
                input.dims()
            )));
        }
        let dims = input.dims().to_vec();
        let (height, width, channels) = match axis_order {
            ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
            ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
        };
        if channels != 3 {
            return Err(invalid_shape(format!(
                "grayscale requires exactly 3 input channels, got {channels}"
            )));
        }

        let output_channels = usize::from(self.num_output_channels);
        let values = input.with_cpu_storage(|storage, layout| {
            let CpuStorageRef::U8(values) = storage else {
                return Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                });
            };
            let stride = layout.stride();
            let read = |coords: [usize; 3]| {
                let offset = coords.iter().zip(stride).try_fold(
                    layout.start_offset(),
                    |offset, (&coord, &stride)| {
                        let delta = coord
                            .checked_mul(stride)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        offset
                            .checked_add(delta)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)
                    },
                )?;
                values
                    .get(offset)
                    .copied()
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            };
            let mut output = Vec::with_capacity(input.elem_count() / 3 * output_channels);
            let luma = |red: u8, green: u8, blue: u8| {
                ((2126 * u32::from(red) + 7152 * u32::from(green) + 722 * u32::from(blue)) / 10_000)
                    as u8
            };
            match axis_order {
                ImageAxisOrder::Hwc => {
                    for y in 0..height {
                        for x in 0..width {
                            let gray = luma(read([y, x, 0])?, read([y, x, 1])?, read([y, x, 2])?);
                            output.extend(std::iter::repeat_n(gray, output_channels));
                        }
                    }
                }
                ImageAxisOrder::Chw => {
                    for channel in 0..output_channels {
                        for y in 0..height {
                            for x in 0..width {
                                let gray =
                                    luma(read([0, y, x])?, read([1, y, x])?, read([2, y, x])?);
                                let _ = channel;
                                output.push(gray);
                            }
                        }
                    }
                }
            }
            Ok(output)
        })?;
        let output_dims = match axis_order {
            ImageAxisOrder::Hwc => vec![height, width, output_channels],
            ImageAxisOrder::Chw => vec![output_channels, height, width],
        };
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image: Tensor::from_vec(values, output_dims, input.device())?,
            label: sample.label,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{BrightnessConfig, ContrastConfig, GrayscaleConfig, HueConfig};
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    fn sample() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec(vec![255u8, 0, 0, 0, 255, 0], [1, 2, 3], &Device::Cpu).unwrap(),
            label: 4,
        })
    }

    #[test]
    fn brightness_and_contrast_materialize_u8_hwc() {
        let input = sample().into_decoded().unwrap().image;
        let output = BrightnessConfig::new(10)
            .apply(sample())
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert_eq!(output.label, 4);
        assert_eq!(
            output.image.to_vec::<u8>().unwrap(),
            [255, 10, 10, 10, 255, 10]
        );
        assert!(!output.image.same_storage(&input));
        assert!(ContrastConfig::new(f32::NAN).validate().is_err());
    }

    #[test]
    fn hue_preserves_shape_and_dtype() {
        let output = HueConfig::new(90)
            .apply(sample())
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
    }

    #[test]
    fn grayscale_supports_hwc_and_chw_channel_contracts() {
        let hwc = GrayscaleConfig::one_channel()
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(hwc.image.dims(), [1, 2, 1]);
        assert_eq!(hwc.image.to_vec::<u8>().unwrap(), [54, 182]);

        let chw_input =
            Tensor::from_vec(vec![255u8, 0, 0, 255, 0, 0], [3, 1, 2], &Device::Cpu).unwrap();
        let chw = GrayscaleConfig::new(3)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: chw_input,
                    label: 0,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(chw.image.dims(), [3, 1, 2]);
        assert_eq!(
            chw.image.to_vec::<u8>().unwrap(),
            [54, 182, 54, 182, 54, 182]
        );
    }
}
