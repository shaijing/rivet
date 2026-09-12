use crate::dataset::source::{Dataset, validate_indices};
use crate::errors::RivetResult;
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

    fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
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
    fn get_wrapper_uses_get_many_semantics() {
        let dataset = MemoryDataset::new(vec![10, 20, 30]);

        assert_eq!(dataset.get(1).unwrap(), 20);
    }
}
