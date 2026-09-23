use crate::dataset::source::{Dataset, validate_indices};
use crate::errors::DataResult;
use std::sync::Arc;

/// An eagerly materialized, read-only dataset.
///
/// The backing allocation is shared between cloned dataset handles. Samples
/// are cloned only when a caller asks for them through [`Dataset::get_many`].
#[derive(Debug)]
pub struct MemoryDataset<T> {
    items: Arc<[T]>,
}

impl<T> MemoryDataset<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items: Arc::from(items.into_boxed_slice()),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T> Dataset for MemoryDataset<T>
where
    T: Clone + Send + Sync,
{
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn capabilities(&self) -> crate::dataset::SourceCapabilities {
        crate::dataset::SourceCapabilities {
            access_pattern: crate::dataset::AccessPattern::RandomAccess,
            batched_reads: true,
            preferred_batch_size: None,
            zero_copy: false,
            parallel_reads: true,
            async_reads: false,
            read_device: None,
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        validate_indices(indices, self.items.len())?;

        Ok(indices
            .iter()
            .map(|&index| self.items[index].clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_many_preserves_order() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert_eq!(dataset.get_many(&[2, 0, 1]).unwrap(), [30, 10, 20]);
    }

    #[test]
    fn get_many_preserves_duplicates() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert_eq!(dataset.get_many(&[2, 0, 2]).unwrap(), [30, 10, 30]);
    }

    #[test]
    fn get_many_empty_returns_empty() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert!(dataset.get_many(&[]).unwrap().is_empty());
    }

    #[test]
    fn get_many_out_of_range_returns_error() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert!(dataset.get_many(&[0, 3]).is_err());
    }

    #[test]
    fn capabilities_describe_immutable_batched_memory_reads_conservatively() {
        let dataset = MemoryDataset::new(vec![1, 2, 3]);
        let capabilities = dataset.capabilities();
        assert_eq!(
            capabilities.access_pattern,
            crate::dataset::AccessPattern::RandomAccess
        );
        assert!(capabilities.batched_reads);
        assert!(capabilities.parallel_reads);
        assert!(!capabilities.zero_copy);
        assert!(!capabilities.async_reads);
        assert_eq!(capabilities.read_device, None);
    }

    #[test]
    fn get_wrapper_uses_get_many_semantics() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert_eq!(dataset.get(1).unwrap(), 20);
    }
}
