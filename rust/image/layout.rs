use crate::errors::{RivetResult, invalid_shape};
use crate::sample::{DecodedSample, ImageBuffer, ImageLayout, ImageSample};

#[derive(Clone, Copy)]
pub(crate) struct LayoutConfig {
    pub(crate) layout: ImageLayout,
}

impl LayoutConfig {
    pub(crate) fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
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

fn convert_hwc_to_chw(sample: DecodedSample) -> RivetResult<DecodedSample> {
    let DecodedSample {
        image,
        width,
        height,
        channels,
        label,
        layout: _,
    } = sample;

    let out = match image {
        ImageBuffer::U8(values) => ImageBuffer::U8(convert_hwc_values_to_chw(
            values,
            width as usize,
            height as usize,
            channels as usize,
        )?),
        ImageBuffer::F32(values) => ImageBuffer::F32(convert_hwc_values_to_chw(
            values,
            width as usize,
            height as usize,
            channels as usize,
        )?),
    };

    Ok(DecodedSample {
        image: out,
        width,
        height,
        channels,
        label,
        layout: ImageLayout::Chw,
    })
}

fn convert_chw_to_hwc(sample: DecodedSample) -> RivetResult<DecodedSample> {
    let DecodedSample {
        image,
        width,
        height,
        channels,
        label,
        layout: _,
    } = sample;

    let out = match image {
        ImageBuffer::U8(values) => ImageBuffer::U8(convert_chw_values_to_hwc(
            values,
            width as usize,
            height as usize,
            channels as usize,
        )?),
        ImageBuffer::F32(values) => ImageBuffer::F32(convert_chw_values_to_hwc(
            values,
            width as usize,
            height as usize,
            channels as usize,
        )?),
    };

    Ok(DecodedSample {
        image: out,
        width,
        height,
        channels,
        label,
        layout: ImageLayout::Hwc,
    })
}

fn convert_hwc_values_to_chw<T: Copy>(
    values: Vec<T>,
    width: usize,
    height: usize,
    channels: usize,
) -> RivetResult<Vec<T>> {
    let expected = width * height * channels;

    if values.len() != expected {
        return Err(invalid_shape(
            "image buffer length does not match HWC shape",
        ));
    }

    let mut out = Vec::with_capacity(values.len());
    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                out.push(values[(h * width + w) * channels + c]);
            }
        }
    }

    Ok(out)
}

fn convert_chw_values_to_hwc<T: Copy>(
    values: Vec<T>,
    width: usize,
    height: usize,
    channels: usize,
) -> RivetResult<Vec<T>> {
    let expected = width * height * channels;

    if values.len() != expected {
        return Err(invalid_shape(
            "image buffer length does not match CHW shape",
        ));
    }

    let mut out = Vec::with_capacity(values.len());
    for h in 0..height {
        for w in 0..width {
            for c in 0..channels {
                out.push(values[c * height * width + h * width + w]);
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::LayoutConfig;
    use crate::sample::{DecodedSample, ImageBuffer, ImageLayout, ImageSample};

    #[test]
    fn converts_hwc_to_chw() {
        let sample = ImageSample::Decoded(DecodedSample {
            image: ImageBuffer::U8(vec![1, 2, 3, 4, 5, 6]),
            width: 2,
            height: 1,
            channels: 3,
            label: 0,
            layout: ImageLayout::Hwc,
        });
        let out = LayoutConfig {
            layout: ImageLayout::Chw,
        }
        .apply(sample)
        .unwrap()
        .into_decoded()
        .unwrap();

        match out.image {
            ImageBuffer::U8(values) => assert_eq!(values, vec![1, 4, 2, 5, 3, 6]),
            ImageBuffer::F32(_) => panic!("expected u8 output"),
        }
    }
}
