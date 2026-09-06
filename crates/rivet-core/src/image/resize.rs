use crate::errors::RivetResult;
use crate::image::{from_rgb_image, into_rgb_image};
use crate::sample::ImageSample;
use image::imageops::{FilterType, resize};

#[derive(Clone, Copy)]
pub struct ResizeConfig {
    pub width: u32,
    pub height: u32,
}

impl ResizeConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "resize")?;
        let resized = resize(&image, self.width, self.height, FilterType::Triangle);
        Ok(ImageSample::Decoded(from_rgb_image(resized, label)))
    }
}
