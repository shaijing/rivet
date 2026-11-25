//! Structured detection/segmentation samples, isolated from the classification
//! fast path. All instance fields share the first dimension.
mod annotations;
mod bbox_safe;
mod bbox_subset;
pub use bbox_safe::BBoxSafeRandomCropConfig;
pub use bbox_subset::BBoxSubsetSafeRandomCropConfig;
mod geometry;
mod mask_crop;
pub use mask_crop::CropNonEmptyMaskConfig;
mod pipeline;
pub use annotations::{
    AtLeastOneBBoxRandomCropConfig, BoxFormat, RandomIoUCropConfig, StructuredSample,
};
pub use pipeline::{StructuredBatch, StructuredLoader, StructuredOp, StructuredPipeline};
