//! Modality-neutral in-memory materialization.

mod policy;

use rivet_data::dataset::memory::MemoryDataset;
use rivet_data::dataset::source::Dataset;
use rivet_data::errors::{DataResult, invalid_argument};

pub use policy::{CacheConfig, CacheLevel, CachePolicy, DEFAULT_CHUNK_SIZE};

/// Materialize any dataset in bounded logical batches.
pub fn materialize_to_memory<T>(
    dataset: &dyn Dataset<Item = T>,
    chunk_size: usize,
) -> DataResult<MemoryDataset<T>>
where
    T: Clone + Send + Sync,
{
    if chunk_size == 0 {
        return Err(invalid_argument("cache chunk_size must be greater than 0"));
    }

    let len = dataset.len();
    let mut items = Vec::with_capacity(len);
    for start in (0..len).step_by(chunk_size) {
        let end = start.saturating_add(chunk_size).min(len);
        let indices = (start..end).collect::<Vec<_>>();
        let chunk = dataset.get_many(&indices)?;
        if chunk.len() != end - start {
            return Err(invalid_argument(format!(
                "dataset returned {} rows for cache chunk of {} indices",
                chunk.len(),
                end - start
            )));
        }
        items.extend(chunk);
    }
    Ok(MemoryDataset::new(items))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct RecordingDataset {
        len: usize,
        requests: Arc<Mutex<Vec<Vec<usize>>>>,
    }

    impl Dataset for RecordingDataset {
        type Item = usize;

        fn len(&self) -> usize {
            self.len
        }

        fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
            self.requests.lock().unwrap().push(indices.to_vec());
            Ok(indices.to_vec())
        }
    }

    #[test]
    fn materializes_in_bounded_batches() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let dataset = RecordingDataset {
            len: 10,
            requests: Arc::clone(&requests),
        };

        let cached = materialize_to_memory(&dataset, 4).unwrap();

        assert_eq!(cached.as_slice(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            *requests.lock().unwrap(),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
        );
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let dataset = rivet_data::dataset::MemoryDataset::new(vec![1usize]);

        assert!(
            materialize_to_memory(&dataset, 0)
                .unwrap_err()
                .to_string()
                .contains("greater than 0")
        );
    }
}
