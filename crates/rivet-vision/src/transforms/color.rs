use crate::errors::RivetResult;
use crate::sample::image::ImageSample;
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::{brighten, contrast};

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

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "contrast")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            contrast(&image, self.value),
            label,
        )?))
    }
}
