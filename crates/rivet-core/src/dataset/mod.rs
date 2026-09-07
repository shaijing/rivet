//! Storage backends (arrow, filesystem, ...) stay modality-free; modality
//! adapters map raw storage rows onto typed samples.

pub mod arrow;
pub mod filesystem;
pub mod source;

pub use arrow::ArrowImageDataset;
pub use filesystem::{ImageFolderDatasetCore, ImageFolderSample};
pub use source::{Dataset, Source};
