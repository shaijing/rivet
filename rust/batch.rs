use crate::errors::value_err;
use crate::sample::{DecodedSample, ImageBatch};
use pyo3::prelude::*;

pub(crate) struct ImageBatchBuilder {
    images: Vec<u8>,
    labels: Vec<i64>,
    expected_shape: Option<(u32, u32, u8)>,
}

impl ImageBatchBuilder {
    pub(crate) fn with_capacity(batch_size: usize) -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::with_capacity(batch_size),
            expected_shape: None,
        }
    }

    pub(crate) fn push(&mut self, sample: DecodedSample) -> PyResult<()> {
        let shape = (sample.height, sample.width, sample.channels);

        match self.expected_shape {
            Some(expected) if expected != shape => Err(value_err(format!(
                "all images in a batch must have the same shape; expected {:?}, got {:?}",
                expected, shape
            ))),
            Some(_) => {
                self.images.extend_from_slice(&sample.image);
                self.labels.push(sample.label);
                Ok(())
            }
            None => {
                self.expected_shape = Some(shape);
                self.images.extend_from_slice(&sample.image);
                self.labels.push(sample.label);
                Ok(())
            }
        }
    }

    pub(crate) fn finish(self) -> ImageBatch {
        let batch_len = self.labels.len();
        let (height, width, channels) = self.expected_shape.unwrap_or((0, 0, 3));

        ImageBatch {
            images: self.images,
            labels: self.labels,
            shape: (
                batch_len,
                height as usize,
                width as usize,
                channels as usize,
            ),
        }
    }
}
