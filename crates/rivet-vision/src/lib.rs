pub mod api;
pub mod batch;
pub mod cache;
pub mod datasets;
pub mod errors;
pub mod pipeline;
pub mod runtime;
pub mod sample;
pub mod source;
pub mod transforms;

pub mod sampler {
    pub(crate) use rivet_data::sampler::permute;
    pub use rivet_data::sampler::{IndexSampler, SamplerPlan};
}

pub use batch::ImageBatchBuilder;
pub use cache::{DecodedImageMemoryDataset, DenseImageMemoryDataset, VariableImageMemoryDataset};
pub use errors::{RivetError, RivetResult, VisionError, VisionResult};
pub use pipeline::{Compose, ImagePipeline, ImageTransform, TransformSequence};
pub use rivet_data::random::{
    OpKey, RNG_ALGORITHM_VERSION, RandomContext, RandomDomain, RandomStream, SampleKey,
};
pub use runtime::ImageDataLoader;
pub use sample::image::{
    DecodedSample, EncodedImageSample, ImageAxisOrder, ImageBatch, ImageSample,
};
pub use source::ImageSource;

/// Stable planning domain identity used by vision payloads and value
/// properties. Other domain crates must use distinct IDs.
pub const VISION_DOMAIN_ID: rivet_plan::DomainId = rivet_plan::DomainId::new(1);
