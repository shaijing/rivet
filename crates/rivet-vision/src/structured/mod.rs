//! Structured detection/segmentation samples, isolated from the classification
//! fast path. All instance fields share the first dimension.
mod annotations;
mod geometry;
mod pipeline;
pub use annotations::{BoxFormat, RandomIoUCropConfig, StructuredSample};
pub use pipeline::{StructuredBatch, StructuredLoader, StructuredOp, StructuredPipeline};
