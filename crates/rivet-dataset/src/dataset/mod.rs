//! Storage backends (arrow, filesystem, ...) stay modality-free; modality
//! adapters map raw storage rows onto typed samples.

pub mod arrow;
pub mod bundle;
pub mod cache;
pub mod filesystem;
pub mod image_source;
pub mod lance;
pub mod manifest;
pub mod memory;
pub mod source;

pub use arrow::{ArrowImageDataset, ArrowTextDataset};
pub use bundle::{DatasetBundle, DatasetLoadResult};
pub use cache::{
    CacheConfig, CacheLevel, CachePolicy, DEFAULT_DECODED_CHUNK_SIZE, DEFAULT_ENCODED_CHUNK_SIZE,
    materialize_decoded_to_memory, materialize_to_memory,
};
pub use filesystem::{ImageFolderDatasetCore, ImageFolderSample};
pub use image_source::ImageSource;
pub use lance::{LanceImageDataset, LanceTable, load_lance_image_dataset};
pub use manifest::{DatasetManifest, SplitManifest};
pub use memory::{DecodedImageMemoryDataset, MemoryDataset};
pub use source::{Dataset, Source};
