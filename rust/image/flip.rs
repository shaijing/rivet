use crate::errors::RivetResult;
use crate::image::{from_rgb_image, into_rgb_image};
use crate::sample::ImageSample;
use image::imageops::{flip_horizontal, flip_vertical};

#[derive(Clone, Copy)]
pub(crate) enum FlipDirection {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy)]
pub(crate) struct FlipConfig {
    pub(crate) direction: FlipDirection,
}

impl FlipConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "flip")?;
        let flipped = match self.direction {
            FlipDirection::Horizontal => flip_horizontal(&image),
            FlipDirection::Vertical => flip_vertical(&image),
        };
        Ok(ImageSample::Decoded(from_rgb_image(flipped, label)))
    }
}
