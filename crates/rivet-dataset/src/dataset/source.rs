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

    /// Fetch one logical batch of rows.
    ///
    /// Implementations must preserve the input order and multiplicity. An
    /// empty request is valid and must not perform storage I/O.
    fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>>;

    /// Convenience wrapper around the batch primitive.
    fn get(&self, index: usize) -> RivetResult<Self::Item> {
        let mut items = self.get_many(&[index])?;
        debug_assert_eq!(items.len(), 1);
        Ok(items.remove(0))
    }

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
        Self { inner: dataset }
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

    pub fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<T>> {
        self.inner.get_many(indices)
    }
}

/// Validate all indices before a backend starts reading storage. Keeping this
/// check shared makes empty requests, bounds errors, and cardinality
/// guarantees consistent across all dataset implementations.
pub(crate) fn validate_indices(indices: &[usize], len: usize) -> RivetResult<()> {
    for &index in indices {
        if index >= len {
            return Err(crate::errors::RivetError::IndexOutOfRange { index, len });
        }
    }
    Ok(())
}
