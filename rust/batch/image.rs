use crate::errors::value_err;
use crate::sample::{DecodedSample, ImageBatch, ImageDType, ImageLayout};
use pyo3::prelude::*;

pub(crate) struct ImageBatchBuilder {
    images: Vec<u8>,
    labels: Vec<i64>,
    expected_shape: Option<(u32, u32, u8)>,
    dtype: Option<ImageDType>,
    layout: Option<ImageLayout>,
}

impl ImageBatchBuilder {
    pub(crate) fn with_capacity(batch_size: usize) -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::with_capacity(batch_size),
            expected_shape: None,
            dtype: None,
            layout: None,
        }
    }

    pub(crate) fn push(&mut self, sample: DecodedSample) -> PyResult<()> {
        let shape = (sample.height, sample.width, sample.channels);

        if let Some(expected) = self.expected_shape {
            if expected != shape {
                return Err(value_err(format!(
                    "all images in a batch must have the same shape; expected {:?}, got {:?}",
                    expected, shape
                )));
            }
        } else {
            self.expected_shape = Some(shape);
        }

        if let Some(dtype) = self.dtype {
            if dtype != sample.dtype {
                return Err(value_err("all images in a batch must have the same dtype"));
            }
        } else {
            self.dtype = Some(sample.dtype);
        }

        if let Some(layout) = self.layout {
            if layout != sample.layout {
                return Err(value_err("all images in a batch must have the same layout"));
            }
        } else {
            self.layout = Some(sample.layout);
        }

        self.images.extend_from_slice(&sample.image);
        self.labels.push(sample.label);
        Ok(())
    }

    pub(crate) fn finish(self) -> ImageBatch {
        let batch_len = self.labels.len();
        let (height, width, channels) = self.expected_shape.unwrap_or((0, 0, 3));
        let dtype = self.dtype.unwrap_or(ImageDType::U8);
        let layout = self.layout.unwrap_or(ImageLayout::Hwc);
        let shape = match layout {
            ImageLayout::Hwc => (
                batch_len,
                height as usize,
                width as usize,
                channels as usize,
            ),
            ImageLayout::Chw => (
                batch_len,
                channels as usize,
                height as usize,
                width as usize,
            ),
        };

        ImageBatch {
            images: self.images,
            labels: self.labels,
            shape,
            dtype,
            layout,
        }
    }
}
