use crate::errors::{DataError, DataResult, invalid_argument};
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
    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>>;

    /// Convenience wrapper around the batch primitive.
    fn get(&self, index: usize) -> DataResult<Self::Item> {
        let items = self.get_many(&[index])?;
        if items.len() != 1 {
            return Err(invalid_argument(format!(
                "dataset returned {} rows for one requested index",
                items.len()
            )));
        }
        Ok(items.into_iter().next().expect("validated one row"))
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shared runtime handle to any [`Dataset`] yielding `T`. Modality aliases
/// the item type, so the source layer itself stays modality-free. A modality
/// crate can wrap this handle when it needs representation-aware state:
///
/// ```text
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

    pub fn get(&self, index: usize) -> DataResult<T> {
        self.inner.get(index)
    }

    pub fn get_many(&self, indices: &[usize]) -> DataResult<Vec<T>> {
        self.inner.get_many(indices)
    }
}

/// Validate all indices before a backend starts reading storage. Keeping this
/// check shared makes empty requests, bounds errors, and cardinality
/// guarantees consistent across all dataset implementations.
pub fn validate_indices(indices: &[usize], len: usize) -> DataResult<()> {
    for &index in indices {
        if index >= len {
            return Err(DataError::IndexOutOfRange { index, len });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::memory::MemoryDataset;
    use std::sync::Arc;

    #[test]
    fn get_wrapper_uses_get_many_semantics() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert_eq!(dataset.get(1).unwrap(), 20);
        assert!(dataset.get(3).is_err());
    }

    #[test]
    fn source_clones_share_the_same_backend_handle() {
        let source = Source::new(Arc::new(MemoryDataset::new(vec![1, 2, 3])));
        let clone = source.clone();

        assert_eq!(source.len(), clone.len());
        assert_eq!(clone.get_many(&[2, 0]).unwrap(), [3, 1]);
    }
}
