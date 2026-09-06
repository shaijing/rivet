use crate::errors::{runtime_err, value_err};
use crate::image::from_rgb_image;
use crate::sample::{DecodedSample, ImageSample};
use pyo3::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct DecodeImageConfig;

impl DecodeImageConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> PyResult<ImageSample> {
        match sample {
            ImageSample::Encoded(sample) => {
                let image = image::load_from_memory(&sample.image)
                    .map_err(runtime_err)?
                    .to_rgb8();
                Ok(ImageSample::Decoded(from_rgb_image(image, sample.label)))
            }
            ImageSample::Decoded(_) => Err(value_err("decode_image received a decoded sample")),
        }
    }
}

pub(crate) fn decode_rgb(encoded: &[u8], label: i64) -> PyResult<DecodedSample> {
    let image = image::load_from_memory(encoded)
        .map_err(runtime_err)?
        .to_rgb8();
    Ok(from_rgb_image(image, label))
}
