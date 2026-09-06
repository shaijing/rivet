use crate::errors::{RivetResult, invalid_shape};
use crate::image::{from_rgb_image, into_rgb_image};
use crate::sample::ImageSample;
use image::imageops::crop_imm;

#[derive(Clone, Copy)]
pub(crate) struct CropConfig {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl CropConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;

        if self.x + self.width > sample.width || self.y + self.height > sample.height {
            return Err(invalid_shape(format!(
                "crop rectangle ({}, {}, {}, {}) exceeds image shape {}x{}",
                self.x, self.y, self.width, self.height, sample.width, sample.height
            )));
        }

        let (image, label) = into_rgb_image(sample, "crop")?;
        let cropped = crop_imm(&image, self.x, self.y, self.width, self.height).to_image();
        Ok(ImageSample::Decoded(from_rgb_image(cropped, label)))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CenterCropConfig {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl CenterCropConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;

        if self.width > sample.width || self.height > sample.height {
            return Err(invalid_shape(format!(
                "center_crop size {}x{} exceeds image shape {}x{}",
                self.width, self.height, sample.width, sample.height
            )));
        }

        let x = (sample.width - self.width) / 2;
        let y = (sample.height - self.height) / 2;
        CropConfig {
            x,
            y,
            width: self.width,
            height: self.height,
        }
        .apply(ImageSample::Decoded(sample))
    }
}
