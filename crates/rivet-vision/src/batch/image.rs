use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageBatch};
use rivet_core::{DType, Device, Tensor};

/// Builds a batch without forcing individual samples contiguous. `Tensor::stack`
/// reads each sample in logical order and allocates the final batch once.
pub struct ImageBatchBuilder {
    images: Vec<Tensor>,
    labels: Vec<i64>,
    expected_shape: Option<Vec<usize>>,
    expected_dtype: Option<DType>,
}

impl ImageBatchBuilder {
    pub fn with_capacity(batch_size: usize) -> Self {
        Self {
            images: Vec::with_capacity(batch_size),
            labels: Vec::with_capacity(batch_size),
            expected_shape: None,
            expected_dtype: None,
        }
    }

    pub(crate) fn with_reusable_labels(batch_size: usize, mut labels: Vec<i64>) -> Self {
        labels.clear();
        labels.reserve(batch_size);
        Self {
            images: Vec::with_capacity(batch_size),
            labels,
            expected_shape: None,
            expected_dtype: None,
        }
    }

    pub fn push(&mut self, sample: DecodedSample) -> RivetResult<()> {
        let shape = sample.image.dims();
        if let Some(expected) = &self.expected_shape {
            if expected.as_slice() != shape {
                return Err(invalid_shape(format!(
                    "all images in a batch must have the same shape; expected {:?}, got {:?}",
                    expected, shape
                )));
            }
        } else {
            self.expected_shape = Some(shape.to_vec());
        }

        if let Some(expected) = self.expected_dtype {
            if expected != sample.image.dtype() {
                return Err(invalid_argument(format!(
                    "all images in a batch must have the same dtype; expected {:?}, got {:?}",
                    expected,
                    sample.image.dtype()
                )));
            }
        } else {
            self.expected_dtype = Some(sample.image.dtype());
        }

        self.images.push(sample.image);
        self.labels.push(sample.label);
        Ok(())
    }

    pub fn finish(self) -> RivetResult<ImageBatch> {
        let labels = Tensor::from_vec(self.labels, [self.images.len()], &Device::Cpu)?;
        Self::finish_images(self.images, labels)
    }

    pub(crate) fn finish_reusing_labels(self, staging: &mut Vec<i64>) -> RivetResult<ImageBatch> {
        // Tensor construction copies into independent aligned storage. The
        // staging vector remains reusable even while prior batches are live.
        let labels = Tensor::from_slice(&self.labels, [self.images.len()], &Device::Cpu);
        *staging = self.labels;
        staging.clear();
        Self::finish_images(self.images, labels?)
    }

    fn finish_images(samples: Vec<Tensor>, labels: Tensor) -> RivetResult<ImageBatch> {
        let images = if samples.is_empty() {
            Tensor::from_vec(Vec::<u8>::new(), [0usize], &Device::Cpu)?
        } else {
            let refs: Vec<&Tensor> = samples.iter().collect();
            Tensor::stack(&refs, 0)?
        };

        Ok(ImageBatch {
            images,
            labels,
            axis_order: ImageAxisOrder::Hwc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ImageBatchBuilder;
    use crate::sample::image::DecodedSample;
    use rivet_core::{DType, Device, Tensor};

    fn sample(values: Vec<f32>) -> DecodedSample {
        DecodedSample {
            image: Tensor::from_vec(values, [1, 1, 1], &Device::Cpu).unwrap(),
            label: 7,
        }
    }

    #[test]
    fn label_workspace_is_reused_while_published_batches_remain_live() {
        let mut workspace = Vec::with_capacity(2);
        let workspace_ptr = workspace.as_ptr();
        let mut first = ImageBatchBuilder::with_reusable_labels(2, std::mem::take(&mut workspace));
        first.push(sample(vec![1.0])).unwrap();
        let first = first.finish_reusing_labels(&mut workspace).unwrap();
        assert_eq!(workspace.as_ptr(), workspace_ptr);
        assert_ne!(first.labels.storage_base_ptr(), workspace_ptr.cast());
        let published = first.labels.clone();
        let mut second = ImageBatchBuilder::with_reusable_labels(2, std::mem::take(&mut workspace));
        let mut changed = sample(vec![2.0]);
        changed.label = 12;
        second.push(changed).unwrap();
        let second = second.finish_reusing_labels(&mut workspace).unwrap();
        assert_eq!(workspace.as_ptr(), workspace_ptr);
        assert_eq!(published.to_vec::<i64>().unwrap(), [7]);
        assert_eq!(second.labels.to_vec::<i64>().unwrap(), [12]);
    }

    #[test]
    fn batch_builder_stacks_images_and_labels() {
        let mut builder = ImageBatchBuilder::with_capacity(2);
        builder.push(sample(vec![1.0])).unwrap();
        builder.push(sample(vec![2.0])).unwrap();
        let batch = builder.finish().unwrap();

        assert_eq!(batch.images.dims(), [2, 1, 1, 1]);
        assert_eq!(batch.images.dtype(), DType::F32);
        assert_eq!(batch.images.to_vec::<f32>().unwrap(), [1.0, 2.0]);
        assert_eq!(batch.labels.dims(), [2]);
        assert_eq!(batch.labels.to_vec::<i64>().unwrap(), [7, 7]);
    }
}
