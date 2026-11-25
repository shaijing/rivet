//! Structured detection/segmentation samples, isolated from the classification
//! fast path. All instance fields share the first dimension.
mod annotations;
mod bbox_safe;
pub use bbox_safe::BBoxSafeRandomCropConfig;
mod geometry;
mod pipeline;
pub use annotations::{
    AtLeastOneBBoxRandomCropConfig, BoxFormat, RandomIoUCropConfig, StructuredSample,
};
pub use pipeline::{StructuredBatch, StructuredLoader, StructuredOp, StructuredPipeline};
