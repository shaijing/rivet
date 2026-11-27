pub mod arrow;
pub mod bundle;
pub mod catalog;
pub mod download;
pub mod manifest;
pub mod memory;
mod prepare;
pub mod source;

#[cfg(feature = "lance")]
pub mod lance;

pub use prepare::{PreparedFormat, PreparedWriter, convert_files};

pub use bundle::{DatasetBundle, DatasetLoadResult};
pub use catalog::{DatasetEntry, delete_dataset, list_datasets};
pub use download::{
    DataFiles, DownloadFormat, DownloadOptions, TorchDatasetProvider, dataset_cache_dir,
    download_dataset, download_dataset_with_torch, is_remote_dataset, prepare_downloaded_path,
};
pub use manifest::{
    ClassLabelFeature, DATASET_ORIGIN_KEY, DatasetManifest, Feature, ImageFeature,
    ImageRepresentation, LEGACY_MANIFEST_FILE_NAME, MANIFEST_FILE_NAME, SplitManifest,
    manifest_path,
};
pub use memory::MemoryDataset;
pub use source::{AccessPattern, Dataset, Source, SourceCapabilities};

/// Transitional aliases for the pre-P9 dataset namespace.
///
/// Arrow-backed implementation types are canonical under [`arrow`]. Keep
/// these aliases for one migration window so downstream crates can move
/// without an abrupt break; new code should use `dataset::arrow::{...}`.
#[deprecated(note = "use rivet_data::dataset::arrow::ArrowRow instead")]
pub use arrow::ArrowRow;
#[deprecated(note = "use rivet_data::dataset::arrow::MmapArrowTable instead")]
pub use arrow::MmapArrowTable;
