pub(crate) mod color;
pub(crate) mod crop;
pub(crate) mod decode;
pub(crate) mod flip;
pub(crate) mod layout;
pub(crate) mod normalize;
pub(crate) mod resize;

use crate::errors::value_err;
use crate::sample::{DecodedSample, ImageDType, ImageLayout};
use image::RgbImage;
use pyo3::prelude::*;

pub(crate) fn require_u8_hwc(sample: DecodedSample, op_name: &str) -> PyResult<DecodedSample> {
    if sample.dtype != ImageDType::U8 || sample.layout != ImageLayout::Hwc {
        return Err(value_err(format!(
            "{op_name} requires uint8 HWC input, got {} {}",
            sample.dtype.as_str(),
            sample.layout.as_str()
        )));
    }

    Ok(sample)
}

pub(crate) fn into_rgb_image(sample: DecodedSample, op_name: &str) -> PyResult<(RgbImage, i64)> {
    let sample = require_u8_hwc(sample, op_name)?;
    let label = sample.label;
    let image = RgbImage::from_raw(sample.width, sample.height, sample.image).ok_or_else(|| {
        value_err(format!(
            "{op_name} received invalid image buffer for shape {}x{}x{}",
            sample.width, sample.height, sample.channels
        ))
    })?;

    Ok((image, label))
}

pub(crate) fn from_rgb_image(image: RgbImage, label: i64) -> DecodedSample {
    let (width, height) = image.dimensions();

    DecodedSample {
        image: image.into_raw(),
        width,
        height,
        channels: 3,
        label,
        dtype: ImageDType::U8,
        layout: ImageLayout::Hwc,
    }
}
