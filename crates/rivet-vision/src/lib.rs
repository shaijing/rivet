pub mod api;
pub mod batch;
pub mod cache;
pub mod datasets;
pub mod errors;
#[deprecated(note = "use rivet_vision::transforms or rivet_vision::api instead")]
pub mod image;
pub mod pipeline;
pub mod runtime;
pub mod sample;
pub mod source;
pub mod transforms;

/// Deprecated singular compatibility namespace.
///
/// Use `rivet_vision::datasets`, `rivet_vision::cache`, and
/// `rivet_vision::source` for new code. This shim remains for one migration
/// window so downstream crates can move independently.
#[deprecated(note = "use rivet_vision::datasets, ::cache, or ::source instead")]
pub mod dataset {
    pub use crate::cache::{
        CacheConfig, CacheLevel, CachePolicy, DecodedImageMemoryDataset, DenseImageMemoryDataset,
        VariableImageMemoryDataset,
    };
    pub use crate::datasets::{ArrowImageDataset, ImageFolderDatasetCore, ImageFolderSample};
    #[cfg(feature = "lance")]
    pub use crate::datasets::{LanceImageDataset, load_lance_image_dataset};
    pub use crate::source::ImageSource;
    pub use rivet_data::dataset::{Dataset, DatasetBundle, DatasetLoadResult, Source};
    pub use rivet_data::dataset::{
        DatasetManifest, MANIFEST_FILE_NAME, MemoryDataset, SplitManifest,
    };
}

pub mod sampler {
    pub(crate) use rivet_data::sampler::permute;
    pub use rivet_data::sampler::{IndexSampler, SamplerPlan};
}

pub use batch::ImageBatchBuilder;
pub use cache::{DecodedImageMemoryDataset, DenseImageMemoryDataset, VariableImageMemoryDataset};
pub use errors::{RivetError, RivetResult, VisionError, VisionResult};
pub use pipeline::ImagePipeline;
pub use runtime::ImageDataLoader;
pub use sample::image::{DecodedSample, EncodedImageSample, ImageBatch, ImageLayout, ImageSample};
pub use source::ImageSource;
