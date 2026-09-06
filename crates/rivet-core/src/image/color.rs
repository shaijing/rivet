use crate::errors::RivetResult;
use crate::image::{from_rgb_image, into_rgb_image};
use crate::sample::ImageSample;
use image::imageops::{brighten, contrast};

#[derive(Clone, Copy)]
pub struct BrightnessConfig {
    pub value: i32,
}

impl BrightnessConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "brightness")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            brighten(&image, self.value),
            label,
        )))
    }
}

#[derive(Clone, Copy)]
pub struct ContrastConfig {
    pub value: f32,
}

impl ContrastConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "contrast")?;
        Ok(ImageSample::Decoded(from_rgb_image(
            contrast(&image, self.value),
            label,
        )))
    }
}
