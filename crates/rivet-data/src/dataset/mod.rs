pub mod arrow;
pub mod bundle;
pub mod manifest;
pub mod memory;
pub mod source;

#[cfg(feature = "lance")]
pub mod lance;

pub use bundle::{DatasetBundle, DatasetLoadResult};
pub use manifest::{
    ClassLabelFeature, DatasetManifest, Feature, ImageFeature, ImageRepresentation,
    LEGACY_MANIFEST_FILE_NAME, MANIFEST_FILE_NAME, SplitManifest, manifest_path,
};
pub use memory::MemoryDataset;
pub use source::{Dataset, Source};

/// Transitional aliases for the pre-P9 dataset namespace.
///
/// Arrow-backed implementation types are canonical under [`arrow`]. Keep
/// these aliases for one migration window so downstream crates can move
/// without an abrupt break; new code should use `dataset::arrow::{...}`.
#[deprecated(note = "use rivet_data::dataset::arrow::ArrowRow instead")]
pub use arrow::ArrowRow;
#[deprecated(note = "use rivet_data::dataset::arrow::MmapArrowTable instead")]
pub use arrow::MmapArrowTable;
