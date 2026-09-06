use crate::errors::value_err;
use crate::sample::{DecodedSample, ImageLayout, ImageSample};
use pyo3::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct LayoutConfig {
    pub(crate) layout: ImageLayout,
}

impl LayoutConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> PyResult<ImageSample> {
        let sample = sample.into_decoded()?;

        if sample.layout == self.layout {
            return Ok(ImageSample::Decoded(sample));
        }

        let converted = match (sample.layout, self.layout) {
            (ImageLayout::Hwc, ImageLayout::Chw) => convert_hwc_to_chw(sample)?,
            (ImageLayout::Chw, ImageLayout::Hwc) => convert_chw_to_hwc(sample)?,
            _ => sample,
        };

        Ok(ImageSample::Decoded(converted))
    }
}

fn convert_hwc_to_chw(sample: DecodedSample) -> PyResult<DecodedSample> {
    let bytes_per_value = sample.dtype.bytes_per_value();
    let width = sample.width as usize;
    let height = sample.height as usize;
    let channels = sample.channels as usize;
    let expected = width * height * channels * bytes_per_value;

    if sample.image.len() != expected {
        return Err(value_err("image buffer length does not match HWC shape"));
    }

    let mut out = vec![0u8; sample.image.len()];

    for h in 0..height {
        for w in 0..width {
            for c in 0..channels {
                let src = ((h * width + w) * channels + c) * bytes_per_value;
                let dst = (c * height * width + h * width + w) * bytes_per_value;
                out[dst..dst + bytes_per_value]
                    .copy_from_slice(&sample.image[src..src + bytes_per_value]);
            }
        }
    }

    Ok(with_layout(sample, out, ImageLayout::Chw))
}

fn convert_chw_to_hwc(sample: DecodedSample) -> PyResult<DecodedSample> {
    let bytes_per_value = sample.dtype.bytes_per_value();
    let width = sample.width as usize;
    let height = sample.height as usize;
    let channels = sample.channels as usize;
    let expected = width * height * channels * bytes_per_value;

    if sample.image.len() != expected {
        return Err(value_err("image buffer length does not match CHW shape"));
    }

    let mut out = vec![0u8; sample.image.len()];

    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                let src = (c * height * width + h * width + w) * bytes_per_value;
                let dst = ((h * width + w) * channels + c) * bytes_per_value;
                out[dst..dst + bytes_per_value]
                    .copy_from_slice(&sample.image[src..src + bytes_per_value]);
            }
        }
    }

    Ok(with_layout(sample, out, ImageLayout::Hwc))
}

fn with_layout(mut sample: DecodedSample, image: Vec<u8>, layout: ImageLayout) -> DecodedSample {
    sample.image = image;
    sample.layout = layout;
    sample
}
