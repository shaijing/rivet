//! Storage backends (arrow, filesystem, ...) stay modality-free; modality
//! adapters map raw storage rows onto typed samples.

pub mod arrow;
pub mod bundle;
pub mod cache;
pub mod filesystem;
pub mod lance;
pub mod manifest;
pub mod memory;
pub mod source;

pub use arrow::{ArrowImageDataset, ArrowTextDataset};
pub use bundle::{DatasetBundle, DatasetLoadResult};
pub use cache::{CachePolicy, DEFAULT_ENCODED_CHUNK_SIZE, materialize_to_memory};
pub use filesystem::{ImageFolderDatasetCore, ImageFolderSample};
pub use lance::{LanceImageDataset, LanceTable, load_lance_image_dataset};
pub use manifest::{DatasetManifest, SplitManifest};
pub use memory::MemoryDataset;
pub use source::{Dataset, Source};
