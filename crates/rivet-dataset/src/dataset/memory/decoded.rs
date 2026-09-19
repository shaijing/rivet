use super::MemoryDataset;
use crate::dataset::source::Dataset;
use crate::errors::RivetResult;
use crate::sample::image::DecodedSample;

/// In-memory decoded image samples. Tensor handles share their backing
/// allocation when samples are cloned or requested repeatedly.
#[derive(Debug)]
pub struct DecodedImageMemoryDataset {
    inner: MemoryDataset<DecodedSample>,
}

impl DecodedImageMemoryDataset {
    pub fn new(items: Vec<DecodedSample>) -> Self {
        Self {
            inner: MemoryDataset::new(items),
        }
    }

    pub fn as_slice(&self) -> &[DecodedSample] {
        self.inner.as_slice()
    }
}

impl Dataset for DecodedImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
        self.inner.get_many(indices)
    }
}
