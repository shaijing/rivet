use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::{DecodedSample, ImageBuffer, ImageLayout, ImageSample};

#[derive(Clone)]
pub struct NormalizeConfig {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl NormalizeConfig {
    /// Raw configuration; parameter validation happens at pipeline compile
    /// time via [`NormalizeConfig::validate`].
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.mean.is_empty() || self.std.is_empty() {
            return Err(invalid_argument("normalize mean and std must not be empty"));
        }
        if self.mean.len() != self.std.len() {
            return Err(invalid_argument(
                "normalize mean and std must have the same length",
            ));
        }
        if self.std.iter().any(|value| *value == 0.0) {
            return Err(invalid_argument("normalize std values must be non-zero"));
        }

        Ok(())
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let channel_count = sample.channels as usize;

        if self.mean.len() != 1 && self.mean.len() != channel_count {
            return Err(invalid_argument(format!(
                "normalize mean/std length must be 1 or channel count {}, got {}",
                channel_count,
                self.mean.len()
            )));
        }

        let values = match sample.image {
            ImageBuffer::U8(values) => values
                .iter()
                .map(|value| f32::from(*value) / 255.0)
                .collect::<Vec<_>>(),
            ImageBuffer::F32(values) => values,
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
            image: ImageBuffer::F32(normalized),
            width: sample.width,
            height: sample.height,
            channels: sample.channels,
            label: sample.label,
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

#[cfg(test)]
mod tests {
    use super::NormalizeConfig;
    use crate::sample::image::{DecodedSample, ImageBuffer, ImageLayout, ImageSample};

    #[test]
    fn normalize_u8_to_f32() {
        let sample = ImageSample::Decoded(DecodedSample {
            image: ImageBuffer::U8(vec![0, 255, 128]),
            width: 1,
            height: 1,
            channels: 3,
            label: 0,
            layout: ImageLayout::Hwc,
        });
        let out = NormalizeConfig::new(vec![0.5], vec![0.5])
            .apply(sample)
            .unwrap()
            .into_decoded()
            .unwrap();

        match out.image {
            ImageBuffer::F32(values) => {
                assert_eq!(values[0], -1.0);
                assert_eq!(values[1], 1.0);
            }
            ImageBuffer::U8(_) => panic!("expected f32 output"),
        }
    }
}
