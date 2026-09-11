pub mod color;
pub mod crop;
pub mod decode;
pub mod flip;
pub mod layout;
pub mod normalize;
pub mod resize;

use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageBuffer, ImageDType, ImageLayout};
use image::RgbImage;

pub fn require_u8_hwc(sample: DecodedSample, op_name: &str) -> RivetResult<DecodedSample> {
    if sample.image.dtype() != ImageDType::U8 || sample.layout != ImageLayout::Hwc {
        return Err(invalid_argument(format!(
            "{op_name} requires uint8 HWC input, got {} {}",
            sample.image.dtype().as_str(),
            sample.layout.as_str()
        )));
    }

    Ok(sample)
}

pub fn into_rgb_image(sample: DecodedSample, op_name: &str) -> RivetResult<(RgbImage, i64)> {
    let sample = require_u8_hwc(sample, op_name)?;
    let label = sample.label;
    let ImageBuffer::U8(values) = sample.image else {
        return Err(invalid_argument(format!("{op_name} requires uint8 input")));
    };
    let image = RgbImage::from_raw(sample.width, sample.height, values).ok_or_else(|| {
        invalid_shape(format!(
            "{op_name} received invalid image buffer for shape {}x{}x{}",
            sample.width, sample.height, sample.channels
        ))
    })?;

    Ok((image, label))
}

pub fn from_rgb_image(image: RgbImage, label: i64) -> DecodedSample {
    let (width, height) = image.dimensions();

    DecodedSample {
        image: ImageBuffer::U8(image.into_raw()),
        width,
        height,
        channels: 3,
        label,
        layout: ImageLayout::Hwc,
    }
}
