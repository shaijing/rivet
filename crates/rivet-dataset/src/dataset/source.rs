use crate::errors::RivetResult;
use std::sync::Arc;

/// A modality-agnostic dataset: storage backends implement this once per
/// item type, and typed pipelines consume the associated item.
///
/// `Item: Send` keeps every dataset usable across worker threads once the
/// runtime parallelizes.
pub trait Dataset: Send + Sync {
    type Item: Send;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> RivetResult<Self::Item>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shared runtime handle to any [`Dataset`] yielding `T`. Modality aliases
/// the item type, so the source layer itself stays modality-free:
///
/// ```text
/// ImageSource = Source<EncodedImageSample>
/// TextSource  = Source<RawTextSample>
/// ```
pub struct Source<T> {
    inner: Arc<dyn Dataset<Item = T>>,
}

impl<T> Clone for Source<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Send> Source<T> {
    pub fn new<D>(dataset: Arc<D>) -> Self
    where
        D: Dataset<Item = T> + 'static,
    {
        Self {
            inner: dataset,
        }
    }

    /// Build a source from an already type-erased dataset handle.
    pub fn from_arc(dataset: Arc<dyn Dataset<Item = T>>) -> Self {
        Self { inner: dataset }
    }

    /// Borrow the erased dataset (metadata, future reader factory, ...).
    pub fn as_dataset(&self) -> &dyn Dataset<Item = T> {
        self.inner.as_ref()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn get(&self, index: usize) -> RivetResult<T> {
        self.inner.get(index)
    }
}
