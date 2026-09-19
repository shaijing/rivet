use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::DecodedSample;
use rivet_core::{DType, Device, Tensor};
use rivet_data::dataset::MemoryDataset;
use rivet_data::dataset::source::{Dataset, validate_indices};
use rivet_data::errors::{DataError, DataResult};

/// A fixed-shape decoded image dataset backed by one image Tensor and one
/// label Tensor. Per-sample access only creates views into `images`.
#[derive(Debug)]
pub struct DenseImageMemoryDataset {
    images: Tensor,
    labels: Tensor,
}

impl DenseImageMemoryDataset {
    pub fn new(images: Tensor, labels: Tensor) -> RivetResult<Self> {
        if images.rank() == 0 {
            return Err(invalid_shape(
                "dense image dataset images must have a leading sample dimension",
            ));
        }
        if labels.rank() != 1 {
            return Err(invalid_shape(format!(
                "dense image dataset labels must be rank 1, got shape {:?}",
                labels.dims()
            )));
        }
        if labels.dtype() != DType::I64 {
            return Err(invalid_argument(format!(
                "dense image dataset labels must be int64, got {:?}",
                labels.dtype()
            )));
        }
        if images.dims()[0] != labels.dims()[0] {
            return Err(invalid_shape(format!(
                "dense image dataset length mismatch: images {}, labels {}",
                images.dims()[0],
                labels.dims()[0]
            )));
        }

        Ok(Self { images, labels })
    }

    pub fn images(&self) -> &Tensor {
        &self.images
    }

    pub fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn get_one(&self, index: usize) -> RivetResult<DecodedSample> {
        let image = self.images.narrow(0, index, 1)?.squeeze(0)?;
        let label = self
            .labels
            .narrow(0, index, 1)?
            .squeeze(0)?
            .to_vec0::<i64>()?;
        Ok(DecodedSample { image, label })
    }
}

impl Dataset for DenseImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        self.images.dims()[0]
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        validate_indices(indices, self.len())?;
        indices
            .iter()
            .map(|&index| {
                self.get_one(index)
                    .map_err(|error| DataError::InvalidArgument(error.to_string()))
            })
            .collect()
    }
}

/// Variable-shape decoded image storage. It remains Tensor-backed, but keeps
/// one Tensor handle per sample because a single dense image shape is not
/// available.
#[derive(Debug)]
pub struct VariableImageMemoryDataset {
    inner: MemoryDataset<DecodedSample>,
}

impl VariableImageMemoryDataset {
    pub fn new(items: Vec<DecodedSample>) -> Self {
        Self {
            inner: MemoryDataset::new(items),
        }
    }

    pub fn as_slice(&self) -> &[DecodedSample] {
        self.inner.as_slice()
    }
}

impl Dataset for VariableImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        self.inner.get_many(indices)
    }
}

/// Decoded image cache storage. Fixed-shape samples use dense tensors;
/// variable-resolution samples retain the existing Tensor-handle fallback.
#[derive(Debug)]
pub enum DecodedImageMemoryDataset {
    Dense(DenseImageMemoryDataset),
    Variable(VariableImageMemoryDataset),
}

impl DecodedImageMemoryDataset {
    /// Preserve the explicit sample-oriented constructor for callers that do
    /// not want shape detection. Cache materialization uses `from_samples`.
    pub fn new(items: Vec<DecodedSample>) -> Self {
        Self::Variable(VariableImageMemoryDataset::new(items))
    }

    pub fn from_samples(items: Vec<DecodedSample>) -> RivetResult<Self> {
        let Some(first) = items.first() else {
            return Ok(Self::Variable(VariableImageMemoryDataset::new(items)));
        };

        let shape = first.image.dims();
        let dtype = first.image.dtype();
        let fixed_shape = items
            .iter()
            .all(|sample| sample.image.dims() == shape && sample.image.dtype() == dtype);
        if !fixed_shape {
            return Ok(Self::Variable(VariableImageMemoryDataset::new(items)));
        }

        let labels = items.iter().map(|sample| sample.label).collect::<Vec<_>>();
        let image_refs = items.iter().map(|sample| &sample.image).collect::<Vec<_>>();
        let images = Tensor::stack(&image_refs, 0)?;
        let labels = Tensor::from_vec(labels, [items.len()], &Device::Cpu)?;
        Ok(Self::Dense(DenseImageMemoryDataset::new(images, labels)?))
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    pub fn as_dense(&self) -> Option<&DenseImageMemoryDataset> {
        match self {
            Self::Dense(dataset) => Some(dataset),
            Self::Variable(_) => None,
        }
    }
}

impl Dataset for DecodedImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        match self {
            Self::Dense(dataset) => dataset.len(),
            Self::Variable(dataset) => dataset.len(),
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        match self {
            Self::Dense(dataset) => dataset.get_many(indices),
            Self::Variable(dataset) => dataset.get_many(indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(value: u8, label: i64, shape: [usize; 3]) -> DecodedSample {
        DecodedSample {
            image: Tensor::from_vec(vec![value; shape.iter().product()], shape, &Device::Cpu)
                .unwrap(),
            label,
        }
    }

    #[test]
    fn fixed_shape_samples_use_dense_zero_copy_views() {
        let dataset = DecodedImageMemoryDataset::from_samples(vec![
            sample(10, 10, [1, 2, 1]),
            sample(20, 20, [1, 2, 1]),
            sample(30, 30, [1, 2, 1]),
        ])
        .unwrap();

        assert!(dataset.is_dense());
        let dense = dataset.as_dense().unwrap();
        let samples = dataset.get_many(&[2, 0, 2]).unwrap();
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.label)
                .collect::<Vec<_>>(),
            [30, 10, 30]
        );
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.image.to_vec::<u8>().unwrap()[0])
                .collect::<Vec<_>>(),
            [30, 10, 30]
        );
        assert!(samples[0].image.same_storage(dense.images()));
        assert!(samples[1].image.same_storage(dense.images()));
    }

    #[test]
    fn variable_shapes_keep_tensor_sample_fallback() {
        let dataset = DecodedImageMemoryDataset::from_samples(vec![
            sample(10, 10, [1, 2, 1]),
            sample(20, 20, [2, 2, 1]),
        ])
        .unwrap();

        assert!(!dataset.is_dense());
        let samples = dataset.get_many(&[1, 0, 1]).unwrap();
        assert_eq!(samples[0].image.dims(), [2, 2, 1]);
        assert_eq!(samples[1].image.dims(), [1, 2, 1]);
        assert_eq!(samples[2].label, 20);
    }
}
