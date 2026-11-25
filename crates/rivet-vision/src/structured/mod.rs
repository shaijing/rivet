//! Structured detection/segmentation samples, isolated from the classification
//! fast path. All instance fields share the first dimension.
mod annotations;
mod bbox_safe;
mod bbox_subset;
pub use bbox_safe::BBoxSafeRandomCropConfig;
pub use bbox_subset::BBoxSubsetSafeRandomCropConfig;
mod constrained_dropout;
pub use constrained_dropout::ConstrainedCoarseDropoutConfig;
mod copy_paste;
pub use copy_paste::{CopyAndPasteConfig, CopyPasteDonor};
mod geometry;
mod guided_dropout;
pub use guided_dropout::GuidedCoarseDropoutConfig;
mod mask_crop;
mod mask_dropout;
pub use mask_crop::CropNonEmptyMaskConfig;
pub use mask_dropout::MaskDropoutConfig;
mod mosaic;
pub use mosaic::{MosaicConfig, MosaicFit};
mod near_bbox;
mod overlay;
pub use overlay::{OverlayElement, OverlayElementsConfig};
mod pipeline;
mod spatial_dropout;
pub use annotations::{
    AtLeastOneBBoxRandomCropConfig, BoxFormat, RandomIoUCropConfig, StructuredSample,
};
pub use near_bbox::RandomCropNearBBoxConfig;
pub use pipeline::{StructuredBatch, StructuredLoader, StructuredOp, StructuredPipeline};
pub use spatial_dropout::{SpatialDropout, StructuredDropoutConfig};
