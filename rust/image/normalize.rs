use crate::errors::value_err;
use crate::sample::{DecodedSample, ImageDType, ImageLayout, ImageSample};
use pyo3::prelude::*;

#[derive(Clone)]
pub(crate) struct NormalizeConfig {
    pub(crate) mean: Vec<f32>,
    pub(crate) std: Vec<f32>,
}

impl NormalizeConfig {
    pub(crate) fn new(mean: Vec<f32>, std: Vec<f32>) -> PyResult<Self> {
        if mean.is_empty() || std.is_empty() {
            return Err(value_err("normalize mean and std must not be empty"));
        }
        if mean.len() != std.len() {
            return Err(value_err(
                "normalize mean and std must have the same length",
            ));
        }
        if std.iter().any(|value| *value == 0.0) {
            return Err(value_err("normalize std values must be non-zero"));
        }

        Ok(Self { mean, std })
    }

    pub(crate) fn apply(&self, sample: ImageSample) -> PyResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let channel_count = sample.channels as usize;

        if self.mean.len() != 1 && self.mean.len() != channel_count {
            return Err(value_err(format!(
                "normalize mean/std length must be 1 or channel count {}, got {}",
                channel_count,
                self.mean.len()
            )));
        }

        let values = match sample.dtype {
            ImageDType::U8 => sample
                .image
                .iter()
                .map(|value| f32::from(*value) / 255.0)
                .collect::<Vec<_>>(),
            ImageDType::F32 => bytes_to_f32(&sample.image)?,
        };
        let normalized = match sample.layout {
            ImageLayout::Hwc => normalize_hwc(&values, &self.mean, &self.std, channel_count),
            ImageLayout::Chw => normalize_chw(
                &values,
                &self.mean,
                &self.std,
                channel_count,
                sample.width as usize,
                sample.height as usize,
            ),
        };

        Ok(ImageSample::Decoded(DecodedSample {
            image: f32_to_bytes(&normalized),
            width: sample.width,
            height: sample.height,
            channels: sample.channels,
            label: sample.label,
            dtype: ImageDType::F32,
            layout: sample.layout,
        }))
    }
}

fn normalize_hwc(values: &[f32], mean: &[f32], std: &[f32], channels: usize) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(offset, value)| {
            let channel = offset % channels;
            let stat_index = if mean.len() == 1 { 0 } else { channel };
            (*value - mean[stat_index]) / std[stat_index]
        })
        .collect()
}

fn normalize_chw(
    values: &[f32],
    mean: &[f32],
    std: &[f32],
    channels: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let plane = width * height;

    values
        .iter()
        .enumerate()
        .map(|(offset, value)| {
            let channel = (offset / plane).min(channels - 1);
            let stat_index = if mean.len() == 1 { 0 } else { channel };
            (*value - mean[stat_index]) / std[stat_index]
        })
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> PyResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(value_err(
            "float32 image buffer length must be divisible by 4",
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }

    bytes
}
