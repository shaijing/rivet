//! Dataset materialization and cache policies.

mod policy;

use crate::dataset::memory::MemoryDataset;
use crate::dataset::source::Dataset;
use crate::errors::{RivetResult, invalid_argument};

pub use policy::{CachePolicy, DEFAULT_ENCODED_CHUNK_SIZE};

/// Materialize a dataset into an in-memory dataset in bounded chunks.
///
/// Each chunk is fetched through one `Dataset::get_many` call. This keeps
/// eager loading from constructing a dataset-sized index vector or issuing a
/// single unbounded storage read.
pub fn materialize_to_memory<T>(
    dataset: &dyn Dataset<Item = T>,
    chunk_size: usize,
) -> RivetResult<MemoryDataset<T>>
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
    use crate::dataset::source::Dataset;
    use crate::errors::{RivetError, RivetResult};
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

        fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
            self.requests.lock().unwrap().push(indices.to_vec());
            Ok(indices.to_vec())
        }
    }

    #[test]
    fn materializes_in_chunks_without_per_item_reads() {
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
        let dataset = MemoryDataset::new(vec![1usize]);
        let error = match materialize_to_memory(&dataset, 0) {
            Ok(_) => panic!("zero chunk size should be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("greater than 0"));
    }

    struct ShortReadDataset;

    impl Dataset for ShortReadDataset {
        type Item = usize;

        fn len(&self) -> usize {
            1
        }

        fn get_many(&self, _indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn rejects_a_short_read_instead_of_returning_partial_cache() {
        let error = materialize_to_memory(&ShortReadDataset, 1).unwrap_err();

        assert!(error.to_string().contains("returned 0 rows"));
    }

    struct FailingDataset;

    impl Dataset for FailingDataset {
        type Item = usize;

        fn len(&self) -> usize {
            3
        }

        fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
            if indices.contains(&2) {
                return Err(RivetError::Worker("synthetic source failure".to_owned()));
            }
            Ok(indices.to_vec())
        }
    }

    #[test]
    fn propagates_source_errors() {
        let error = materialize_to_memory(&FailingDataset, 2).unwrap_err();

        assert!(error.to_string().contains("synthetic source failure"));
    }
}
