//! Stable image facade for integrations such as PyO3 bindings.
//!
//! The implementation remains split across `datasets`, `transforms`,
//! `pipeline`, `runtime`, `sample`, and `source`. Consumers that need a
//! supported cross-module entry point should import from this module instead
//! of depending on those implementation paths directly.

pub use crate::batch::ImageBatchBuilder;
pub use crate::cache::{
    CacheConfig, CacheLevel, CachePolicy, DEFAULT_DECODED_CHUNK_SIZE, DEFAULT_DECODED_SLAB_BYTES,
    DEFAULT_ENCODED_CHUNK_SIZE,
};
#[cfg(feature = "lance")]
pub use crate::datasets::lance::{
    LanceConversionReport, LanceImageDataset, convert_imagefolder_dataset, load_lance_image_dataset,
};
pub use crate::datasets::{ArrowImageDataset, ImageFolderDatasetCore, ImageFolderSample};
pub use crate::errors::{
    RivetError, RivetResult, VisionError, VisionResult, invalid_argument, invalid_pipeline,
    invalid_shape,
};
pub use crate::pipeline::op::ImageOp;
pub use crate::pipeline::{
    Compose, ImageConcat, ImageGraphInfo, ImageOptimizationOptions, ImagePipeline, ImageTransform,
    TransformSequence,
};
pub use crate::runtime::{ImageDataLoader, RuntimeConfig};
pub use crate::sample::image::{
    DecodedSample, EncodedImageSample, ImageAxisOrder, ImageBatch, ImageSample,
};
pub use crate::source::ImageSource;
pub use crate::transforms::{
    AffineConfig, ArbitraryRotateConfig, AugMixConfig, AugMixPlan, AugmentOp, AugmentStep,
    AutoAugmentConfig, AutoAugmentPolicy, AutocontrastConfig, BrightnessConfig, CenterCropConfig,
    ColorJitterConfig, ContrastConfig, ConvertImageDtypeConfig, CropConfig, CropRegion,
    DecodeImageConfig, ElasticTransformConfig, EqualizeConfig, EraseConfig, EraseRegion,
    ErasingValue, FiveCropConfig, FlipConfig, FlipDirection, GammaConfig, GaussianBlurConfig,
    GrayscaleConfig, HueConfig, InterpolationMode, InvertConfig, LayoutConfig,
    LinearTransformationConfig, NormalizeConfig, PadConfig, PaddingMode, PerspectiveConfig, Point2,
    PosterizeConfig, RandAugmentConfig, RandomAffineConfig, RandomCropConfig, RandomErasingConfig,
    RandomGaussianBlurConfig, RandomGrayscaleConfig, RandomHorizontalFlipConfig,
    RandomPerspectiveConfig, RandomResizedCropConfig, RandomRotationConfig, ResizeConfig,
    RotateConfig, RotationAngle, SaturationConfig, SharpnessConfig, SolarizeConfig, TenCropConfig,
    TrivialAugmentWideConfig,
};
pub use rivet_core::DType;
pub use rivet_data::random::{
    OpKey, RNG_ALGORITHM_VERSION, RandomContext, RandomDomain, RandomStream, SampleKey,
};

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

pub use crate::transforms::{V2Config, gaussian_noise, shot_noise};

pub use crate::batch::BatchMixConfig;

pub use crate::structured::{
    BoxFormat, RandomIoUCropConfig, StructuredBatch, StructuredLoader, StructuredOp,
    StructuredPipeline, StructuredSample,
};
pub use crate::video::{VideoLayout, VideoSample, uniform_temporal_subsample};
