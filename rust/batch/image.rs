use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::{DecodedSample, ImageBatch, ImageBuffer, ImageDType, ImageLayout};

enum ImageBatchBufferBuilder {
    Empty,
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl ImageBatchBufferBuilder {
    fn dtype(&self) -> Option<ImageDType> {
        match self {
            Self::Empty => None,
            Self::U8(_) => Some(ImageDType::U8),
            Self::F32(_) => Some(ImageDType::F32),
        }
    }

    fn push(&mut self, image: ImageBuffer) -> RivetResult<()> {
        match (std::mem::replace(self, Self::Empty), image) {
            (Self::Empty, ImageBuffer::U8(values)) => {
                *self = Self::U8(values);
                Ok(())
            }
            (Self::Empty, ImageBuffer::F32(values)) => {
                *self = Self::F32(values);
                Ok(())
            }
            (Self::U8(mut buffer), ImageBuffer::U8(values)) => {
                buffer.extend(values);
                *self = Self::U8(buffer);
                Ok(())
            }
            (Self::F32(mut buffer), ImageBuffer::F32(values)) => {
                buffer.extend(values);
                *self = Self::F32(buffer);
                Ok(())
            }
            (current, _) => {
                *self = current;
                Err(invalid_argument(
                    "all images in a batch must have the same dtype",
                ))
            }
        }
    }

    fn finish(self) -> ImageBuffer {
        match self {
            Self::Empty => ImageBuffer::U8(Vec::new()),
            Self::U8(values) => ImageBuffer::U8(values),
            Self::F32(values) => ImageBuffer::F32(values),
        }
    }
}

pub(crate) struct ImageBatchBuilder {
    images: ImageBatchBufferBuilder,
    labels: Vec<i64>,
    expected_shape: Option<(u32, u32, u8)>,
    layout: Option<ImageLayout>,
}

impl ImageBatchBuilder {
    pub(crate) fn with_capacity(batch_size: usize) -> Self {
        Self {
            images: ImageBatchBufferBuilder::Empty,
            labels: Vec::with_capacity(batch_size),
            expected_shape: None,
            layout: None,
        }
    }

    pub(crate) fn push(&mut self, sample: DecodedSample) -> RivetResult<()> {
        let shape = (sample.height, sample.width, sample.channels);

        if let Some(expected) = self.expected_shape {
            if expected != shape {
                return Err(invalid_shape(format!(
                    "all images in a batch must have the same shape; expected {:?}, got {:?}",
                    expected, shape
                )));
            }
        } else {
            self.expected_shape = Some(shape);
        }

        if let Some(dtype) = self.images.dtype() {
            if dtype != sample.image.dtype() {
                return Err(invalid_argument(
                    "all images in a batch must have the same dtype",
                ));
            }
        }

        if let Some(layout) = self.layout {
            if layout != sample.layout {
                return Err(invalid_argument(
                    "all images in a batch must have the same layout",
                ));
            }
        } else {
            self.layout = Some(sample.layout);
        }

        self.images.push(sample.image)?;
        self.labels.push(sample.label);
        Ok(())
    }

    pub(crate) fn finish(self) -> ImageBatch {
        let batch_len = self.labels.len();
        let (height, width, channels) = self.expected_shape.unwrap_or((0, 0, 3));
        let images = self.images.finish();
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
            images,
            labels: self.labels,
            shape,
            layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ImageBatchBuilder;
    use crate::sample::{DecodedSample, ImageBuffer, ImageLayout};

    fn sample(values: Vec<f32>) -> DecodedSample {
        DecodedSample {
            image: ImageBuffer::F32(values),
            width: 1,
            height: 1,
            channels: 1,
            label: 7,
            layout: ImageLayout::Hwc,
        }
    }

    #[test]
    fn batch_builder_preserves_f32_buffer() {
        let mut builder = ImageBatchBuilder::with_capacity(2);

        builder.push(sample(vec![1.0])).unwrap();
        builder.push(sample(vec![2.0])).unwrap();
        let batch = builder.finish();

        assert_eq!(batch.shape, (2, 1, 1, 1));
        assert_eq!(batch.images.dtype().as_str(), "float32");
        match batch.images {
            ImageBuffer::F32(values) => assert_eq!(values, vec![1.0, 2.0]),
            ImageBuffer::U8(_) => panic!("expected f32 batch"),
        }
    }
}
