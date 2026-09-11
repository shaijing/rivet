//! Storage backends (arrow, filesystem, ...) stay modality-free; modality
//! adapters map raw storage rows onto typed samples.

pub mod arrow;
pub mod bundle;
pub mod filesystem;
pub mod lance;
pub mod manifest;
pub mod source;

pub use arrow::{ArrowImageDataset, ArrowTextDataset};
pub use bundle::{DatasetBundle, DatasetLoadResult};
pub use filesystem::{ImageFolderDatasetCore, ImageFolderSample};
pub use lance::{LanceImageDataset, LanceTable, load_lance_image_dataset};
pub use manifest::{DatasetManifest, SplitManifest};
pub use source::{Dataset, Source};
