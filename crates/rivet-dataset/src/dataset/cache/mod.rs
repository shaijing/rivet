//! Dataset materialization and cache policies.

mod policy;

use crate::dataset::memory::{DecodedImageMemoryDataset, MemoryDataset};
use crate::dataset::source::Dataset;
use crate::errors::{RivetResult, invalid_argument};
use crate::image::decode::decode_rgb;

pub use policy::{
    CacheConfig, CacheLevel, CachePolicy, DEFAULT_DECODED_CHUNK_SIZE, DEFAULT_ENCODED_CHUNK_SIZE,
};

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

/// Decode an encoded image dataset into a Tensor-backed in-memory dataset.
///
/// Decoding is deliberately performed after the source batch is fetched and
/// before any pipeline operations run. Fixed-shape decoded samples are
/// consolidated into one dense image Tensor and one label Tensor; variable
/// shapes retain Tensor-backed sample views without copying pixels on access.
pub fn materialize_decoded_to_memory(
    dataset: &dyn Dataset<Item = crate::sample::image::EncodedImageSample>,
    chunk_size: usize,
    max_bytes: Option<usize>,
) -> RivetResult<DecodedImageMemoryDataset> {
    if chunk_size == 0 {
        return Err(invalid_argument("cache chunk_size must be greater than 0"));
    }

    let len = dataset.len();
    let mut items = Vec::with_capacity(len);
    let mut total_bytes = 0usize;

    for start in (0..len).step_by(chunk_size) {
        let end = start.saturating_add(chunk_size).min(len);
        let indices = (start..end).collect::<Vec<_>>();
        let encoded = dataset.get_many(&indices)?;

        if encoded.len() != indices.len() {
            return Err(invalid_argument(format!(
                "dataset returned {} rows for cache chunk of {} indices",
                encoded.len(),
                indices.len()
            )));
        }

        for (index, sample) in indices.into_iter().zip(encoded) {
            let decoded = decode_rgb(sample.image.as_slice(), sample.label).map_err(|error| {
                invalid_argument(format!("failed to decode image at index {index}: {error}"))
            })?;
            let bytes = decoded.image.logical_bytes();
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid_argument("decoded cache byte count overflow"))?;

            if let Some(max_bytes) = max_bytes {
                if total_bytes > max_bytes {
                    return Err(invalid_argument(format!(
                        "decoded cache requires at least {total_bytes} bytes, exceeding max_bytes {max_bytes} at index {index}"
                    )));
                }
            }

            items.push(decoded);
        }
    }

    Ok(DecodedImageMemoryDataset::from_samples(items)?)
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

    #[test]
    fn decoded_materialization_reports_decode_errors_with_index() {
        let dataset = MemoryDataset::new(vec![crate::sample::image::EncodedImageSample {
            image: arrow_buffer::Buffer::from(b"not an image".to_vec()),
            label: 9,
        }]);

        let error = materialize_decoded_to_memory(&dataset, 1, None).unwrap_err();

        assert!(error.to_string().contains("index 0"));
    }

    #[test]
    fn decoded_materialization_enforces_memory_budget() {
        let dataset = MemoryDataset::new(vec![crate::sample::image::EncodedImageSample {
            image: arrow_buffer::Buffer::from(PNG_1X1.to_vec()),
            label: 9,
        }]);

        let error = materialize_decoded_to_memory(&dataset, 1, Some(2)).unwrap_err();

        assert!(error.to_string().contains("max_bytes 2"));
    }

    const PNG_1X1: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
}
