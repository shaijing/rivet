use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::{brighten, contrast, huerotate};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

pub use super::u8_color::{
    AutocontrastConfig, EqualizeConfig, InvertConfig, PosterizeConfig, SharpnessConfig,
    SolarizeConfig,
};

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorJitterConfig {
    pub brightness: i32,
    pub contrast: f32,
    pub hue: i32,
}

impl ColorJitterConfig {
    /// Construct a stochastic jitter policy. Brightness is sampled from
    /// `[-brightness, brightness]`, contrast from
    /// `[max(0, 1-contrast), 1+contrast]`, and hue from
    /// `[-hue, hue]` degrees.
    pub const fn new(brightness: i32, contrast: f32, hue: i32) -> Self {
        Self {
            brightness,
            contrast,
            hue,
        }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.brightness < 0 || self.hue < 0 || !self.contrast.is_finite() || self.contrast < 0.0
        {
            return Err(invalid_argument(
                "color_jitter brightness/hue must be non-negative and contrast must be finite",
            ));
        }
        Ok(())
    }

    pub fn resolve(&self, rng: &mut RandomStream) -> RivetResult<ColorJitterParams> {
        let brightness = sample_i32(rng, -self.brightness, self.brightness)?;
        let contrast_min = (1.0 - self.contrast).max(0.0);
        let contrast = contrast_min + (1.0 + self.contrast - contrast_min) * rng.next_f32();
        let hue = sample_i32(rng, -self.hue, self.hue)?;
        Ok(ColorJitterParams {
            brightness,
            contrast,
            hue,
        })
    }

    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        self.resolve(rng)?.apply(sample)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorJitterParams {
    pub brightness: i32,
    pub contrast: f32,
    pub hue: i32,
}

impl ColorJitterParams {
    pub fn validate(&self) -> RivetResult<()> {
        if !self.contrast.is_finite() {
            return Err(invalid_argument("color_jitter contrast must be finite"));
        }
        Ok(())
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "color_jitter")?;
        let image = brighten(&image, self.brightness);
        let image = contrast(&image, self.contrast);
        let image = huerotate(&image, self.hue);
        Ok(ImageSample::Decoded(from_rgb_image(image, label)?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomGrayscaleConfig {
    pub probability: f64,
    pub num_output_channels: u8,
}

impl RandomGrayscaleConfig {
    pub const fn new(probability: f64) -> Self {
        Self {
            probability,
            num_output_channels: 1,
        }
    }

    pub const fn with_output_channels(mut self, num_output_channels: u8) -> Self {
        self.num_output_channels = num_output_channels;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(invalid_argument(
                "random_grayscale probability must be in [0.0, 1.0]",
            ));
        }
        GrayscaleConfig::new(self.num_output_channels).validate()
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        if !rng.gen_bool(self.probability)? {
            return Ok(sample);
        }
        GrayscaleConfig::new(self.num_output_channels).apply(sample, axis_order)
    }
}

fn sample_i32(rng: &mut RandomStream, min: i32, max: i32) -> RivetResult<i32> {
    if min == max {
        return Ok(min);
    }
    let span = i64::from(max) - i64::from(min) + 1;
    let offset = rng.gen_range_usize(0..span as usize)? as i64;
    Ok((i64::from(min) + offset) as i32)
}

#[cfg(test)]
mod tests {
    use super::{
        BrightnessConfig, ColorJitterConfig, ContrastConfig, GrayscaleConfig, HueConfig,
        RandomGrayscaleConfig,
    };
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};
    use rivet_data::random::{OpKey, RandomContext};

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

    #[test]
    fn color_jitter_is_seeded_and_keeps_rgb_contract() {
        let config = ColorJitterConfig::new(10, 0.25, 20);
        let left_ctx = SampleContext::with_random(3, RandomContext::new(9));
        let right_ctx = SampleContext::with_random(3, RandomContext::new(9));
        let mut left_rng = left_ctx.stream(OpKey::from_parts("ColorJitter", 0));
        let mut right_rng = right_ctx.stream(OpKey::from_parts("ColorJitter", 0));
        let left = config
            .apply(sample(), &mut left_rng)
            .unwrap()
            .into_decoded()
            .unwrap();
        let right = config
            .apply(sample(), &mut right_rng)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(left.image.dims(), [1, 2, 3]);
        assert_eq!(left.image.dtype(), DType::U8);
        assert_eq!(
            left.image.to_vec::<u8>().unwrap(),
            right.image.to_vec::<u8>().unwrap()
        );
    }

    #[test]
    fn random_grayscale_probability_zero_and_one_are_exact() {
        let ctx = SampleContext::new(0);
        let mut rng = ctx.stream(OpKey::from_parts("RandomGrayscale", 0));
        let unchanged = RandomGrayscaleConfig::new(0.0)
            .apply(sample(), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(unchanged.image.dims(), [1, 2, 3]);

        let ctx = SampleContext::new(0);
        let mut rng = ctx.stream(OpKey::from_parts("RandomGrayscale", 0));
        let changed = RandomGrayscaleConfig::new(1.0)
            .apply(sample(), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(changed.image.dims(), [1, 2, 1]);
    }
}
