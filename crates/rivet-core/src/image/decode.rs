use crate::errors::{RivetError, RivetResult, invalid_argument};
use crate::image::from_rgb_image;
use crate::sample::{DecodedSample, ImageSample};

#[derive(Clone, Copy)]
pub struct DecodeImageConfig;

impl DecodeImageConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        match sample {
            ImageSample::Encoded(sample) => {
                let image = image::load_from_memory(&sample.image)
                    .map_err(|err| RivetError::Decode(err.to_string()))?
                    .to_rgb8();
                Ok(ImageSample::Decoded(from_rgb_image(image, sample.label)))
            }
            ImageSample::Decoded(_) => {
                Err(invalid_argument("decode_image received a decoded sample"))
            }
        }
    }
}

pub fn decode_rgb(encoded: &[u8], label: i64) -> RivetResult<DecodedSample> {
    let image = image::load_from_memory(encoded)
        .map_err(|err| RivetError::Decode(err.to_string()))?
        .to_rgb8();
    Ok(from_rgb_image(image, label))
}
