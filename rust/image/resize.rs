use crate::image::{from_rgb_image, into_rgb_image};
use crate::sample::ImageSample;
use image::imageops::{FilterType, resize};
use pyo3::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct ResizeConfig {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl ResizeConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> PyResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "resize")?;
        let resized = resize(&image, self.width, self.height, FilterType::Triangle);
        Ok(ImageSample::Decoded(from_rgb_image(resized, label)))
    }
}
