mod decoded;

use crate::errors::{VisionResult, invalid_argument};
use crate::sample::image::EncodedImageSample;
use crate::transforms::representation::decode_rgb;
use rivet_data::dataset::Dataset;

pub use decoded::{
    DecodedImageMemoryDataset, DenseImageMemoryDataset, PackedImageMemoryDataset,
    VariableImageMemoryDataset,
};

pub const DEFAULT_ENCODED_CHUNK_SIZE: usize = 4096;
pub const DEFAULT_DECODED_CHUNK_SIZE: usize = 4096;
pub const DEFAULT_DECODED_SLAB_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLevel {
    None,
    Encoded,
    Decoded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub level: CacheLevel,
    pub chunk_size: usize,
    pub max_bytes: Option<usize>,
}

impl CacheConfig {
    pub const fn encoded() -> Self {
        Self {
            level: CacheLevel::Encoded,
            chunk_size: DEFAULT_ENCODED_CHUNK_SIZE,
            max_bytes: None,
        }
    }

    pub const fn decoded() -> Self {
        Self {
            level: CacheLevel::Decoded,
            chunk_size: DEFAULT_DECODED_CHUNK_SIZE,
            max_bytes: None,
        }
    }

    pub const fn policy(self) -> CachePolicy {
        match self.level {
            CacheLevel::None => CachePolicy::None,
            CacheLevel::Encoded => CachePolicy::Encoded {
                chunk_size: self.chunk_size,
            },
            CacheLevel::Decoded => CachePolicy::Decoded {
                chunk_size: self.chunk_size,
                max_bytes: self.max_bytes,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    None,
    Encoded {
        chunk_size: usize,
    },
    Decoded {
        chunk_size: usize,
        max_bytes: Option<usize>,
    },
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::None
    }
}

/// Decode an encoded image dataset in bounded source batches and materialize
/// the result into bounded packed slabs. Both fixed- and variable-resolution
/// samples are returned as zero-copy views into those slabs.
pub fn materialize_decoded_to_memory(
    dataset: &dyn Dataset<Item = EncodedImageSample>,
    chunk_size: usize,
    max_bytes: Option<usize>,
) -> VisionResult<DecodedImageMemoryDataset> {
    if chunk_size == 0 {
        return Err(invalid_argument("cache chunk_size must be greater than 0"));
    }

    let mut builder = decoded::PackedImageMemoryBuilder::with_capacity(dataset.len());
    let mut total_bytes = 0usize;
    for start in (0..dataset.len()).step_by(chunk_size) {
        let end = start.saturating_add(chunk_size).min(dataset.len());
        let indices = (start..end).collect::<Vec<_>>();
        let encoded = dataset.get_many(&indices)?;
        if encoded.len() != indices.len() {
            return Err(invalid_argument(format!(
                "dataset returned {} rows for cache chunk of {} indices",
                encoded.len(),
                indices.len()
            )));
        }
        let mut decoded_chunk = Vec::with_capacity(indices.len());
        let mut decoded_chunk_bytes = 0usize;
        for (index, sample) in indices.into_iter().zip(encoded) {
            let decoded = decode_rgb(sample.image.as_slice(), sample.label).map_err(|error| {
                invalid_argument(format!("failed to decode image at index {index}: {error}"))
            })?;
            let decoded_bytes = decoded.image.storage_bytes();
            total_bytes = total_bytes
                .checked_add(decoded_bytes)
                .ok_or_else(|| invalid_argument("decoded cache byte count overflow"))?;
            if let Some(max_bytes) = max_bytes {
                if total_bytes > max_bytes {
                    return Err(invalid_argument(format!(
                        "decoded cache requires at least {total_bytes} bytes, exceeding max_bytes {max_bytes} at index {index}"
                    )));
                }
            }
            if !decoded_chunk.is_empty()
                && decoded_chunk_bytes.saturating_add(decoded_bytes) > DEFAULT_DECODED_SLAB_BYTES
            {
                builder.push_chunk(std::mem::take(&mut decoded_chunk))?;
                decoded_chunk_bytes = 0;
            }
            decoded_chunk_bytes = decoded_chunk_bytes
                .checked_add(decoded_bytes)
                .ok_or_else(|| invalid_argument("decoded cache chunk byte count overflow"))?;
            decoded_chunk.push(decoded);
        }
        builder.push_chunk(decoded_chunk)?;
    }
    Ok(DecodedImageMemoryDataset::Packed(builder.finish()))
}
