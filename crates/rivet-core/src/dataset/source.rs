use crate::errors::RivetResult;
use std::sync::Arc;

/// A modality-agnostic dataset: storage backends implement this once per
/// item type, and typed pipelines consume the associated item.
pub trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> RivetResult<Self::Item>;
}

/// Type-erased view of a [`Dataset`] with a concrete item type.
pub trait DynDataset<T>: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> RivetResult<T>;
}

impl<D, T> DynDataset<T> for D
where
    D: Dataset<Item = T> + Send + Sync,
{
    fn len(&self) -> usize {
        Dataset::len(self)
    }

    fn get(&self, index: usize) -> RivetResult<T> {
        Dataset::get(self, index)
    }
}

/// Shared handle to any [`Dataset`] yielding `T`. Modality aliases the item
/// type, so the source layer itself stays modality-free:
///
/// ```text
/// ImageSource = Source<EncodedImageSample>
/// TextSource  = Source<RawTextSample>
/// ```
pub struct Source<T> {
    inner: Arc<dyn DynDataset<T>>,
}

impl<T> Clone for Source<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Source<T> {
    pub fn new<D>(dataset: Arc<D>) -> Self
    where
        D: Dataset<Item = T> + 'static,
    {
        Self {
            inner: dataset,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn get(&self, index: usize) -> RivetResult<T> {
        self.inner.get(index)
    }
}
