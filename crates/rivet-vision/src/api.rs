//! Stable image facade for integrations such as PyO3 bindings.
//!
//! The implementation remains split across `datasets`, `transforms`,
//! `pipeline`, `runtime`, `sample`, and `source`. Consumers that need a
//! supported cross-module entry point should import from this module instead
//! of depending on those implementation paths directly.

pub use crate::batch::ImageBatchBuilder;
pub use crate::cache::{
    CacheConfig, CacheLevel, CachePolicy, DEFAULT_DECODED_CHUNK_SIZE, DEFAULT_ENCODED_CHUNK_SIZE,
};
pub use crate::datasets::{ArrowImageDataset, ImageFolderDatasetCore, ImageFolderSample};
#[cfg(feature = "lance")]
pub use crate::datasets::{LanceImageDataset, load_lance_image_dataset};
pub use crate::errors::{
    RivetError, RivetResult, VisionError, VisionResult, invalid_argument, invalid_pipeline,
    invalid_shape,
};
pub use crate::pipeline::ImagePipeline;
pub use crate::runtime::{ImageDataLoader, RuntimeConfig};
pub use crate::sample::image::{
    DecodedSample, EncodedImageSample, ImageAxisOrder, ImageBatch, ImageSample,
};
pub use crate::source::ImageSource;
pub use crate::transforms::{
    BrightnessConfig, CenterCropConfig, ContrastConfig, ConvertImageDtypeConfig, CropConfig,
    DecodeImageConfig, FlipConfig, FlipDirection, GrayscaleConfig, HueConfig, InterpolationMode,
    LayoutConfig, NormalizeConfig, PaddingMode, RandomCropConfig, RandomHorizontalFlipConfig,
    ResizeConfig, RotateConfig, RotationAngle,
};
pub use rivet_core::DType;

/// Decode one encoded image and expose it as the standard one-item batch
/// returned by the Python binding.
pub fn decode_image_batch(encoded: &[u8], label: i64) -> VisionResult<ImageBatch> {
    single_sample_batch(crate::transforms::representation::decode_rgb(
        encoded, label,
    )?)
}

/// Turn one decoded sample into the standard image batch representation.
pub fn single_sample_batch(sample: DecodedSample) -> VisionResult<ImageBatch> {
    let mut builder = ImageBatchBuilder::with_capacity(1);
    builder.push(sample)?;
    builder.finish()
}

/// Construct the empty batch used by integration bindings for an exhausted or
/// empty input range.
pub fn empty_image_batch() -> VisionResult<ImageBatch> {
    ImageBatchBuilder::with_capacity(0).finish()
}
