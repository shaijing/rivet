//! Image transform configurations and their native kernels.

mod codecs;
pub mod degradation;
pub mod filtering;
pub mod occlusion;
pub mod photometric;
pub mod policy;
pub mod representation;
pub mod spatial;

pub use degradation::{
    AdditiveDistribution, AdditiveNoiseConfig, AnnotationArtifactsConfig, ArtifactElement,
    AtmosphericFogConfig, FogDepthMode, NoiseSharing, RainType, RandomFogConfig,
    RandomGravelConfig, RandomRainConfig, RandomSnowConfig, SnowMethod, SpatterConfig, SpatterMode,
    gaussian_noise, shot_noise,
};
pub use filtering::{
    DiffusionAlgorithm, DitherMethod, DitheringConfig, EnhanceMode, GaussianBlurConfig,
    LinearTransformationConfig, MorphologicalConfig, MorphologicalOperation, PixelSpreadConfig,
    RandomGaussianBlurConfig, SuperpixelsConfig,
};
pub use occlusion::{
    ChannelDropoutConfig, ChannelDropoutFill, EraseConfig, EraseRegion, ErasingValue,
    GridMaskConfig, PixelDropoutConfig, RandomErasingConfig, StripLength, XyMaskingConfig,
};
pub(crate) use occlusion::{replace_spatial_pixels, sample_coarse_dropout, sample_grid_dropout};
pub use photometric::{
    AutocontrastConfig, BrightnessConfig, ColorJitterConfig, ColorJitterParams, ColorizeConfig,
    ContrastConfig, DistributionMode, EqualizeConfig, FdaConfig, GammaConfig, GrayscaleConfig,
    HEStainConfig, HistogramMatchingConfig, HueConfig, InvertConfig,
    PixelDistributionAdaptationConfig, PosterizeConfig, RandomGrayscaleConfig, ReferenceImages,
    SaturationConfig, SharpnessConfig, SolarizeConfig, StainBasis, StainResidual,
};
pub use policy::{
    AugMixConfig, AugMixPlan, AugmentOp, AugmentStep, AutoAugmentConfig, AutoAugmentPolicy,
    RandAugmentConfig, TrivialAugmentWideConfig,
};
pub use representation::{
    ConvertImageDtypeConfig, DecodeImageConfig, LayoutConfig, NormalizeConfig,
};
pub use spatial::PaddingMode;
pub use spatial::{
    AffineConfig, ArbitraryRotateConfig, CenterCropConfig, CropAndPadConfig, CropConfig,
    CropPadAmounts, CropRegion, ElasticTransformConfig, FiveCropConfig, FlipConfig, FlipDirection,
    GridDistortionConfig, GridElasticDeformConfig, InterpolationMode, LetterBoxConfig,
    LongestMaxSizeConfig, LongestSizeLimit, OpticalDistortionConfig, OpticalDistortionMode,
    PadConfig, PadIfNeededConfig, PadPosition, PerspectiveConfig, PiecewiseAffineConfig, Point2,
    RandomAffineConfig, RandomCropConfig, RandomGridShuffleConfig, RandomHorizontalFlipConfig,
    RandomPerspectiveConfig, RandomResizedCropConfig, RandomRotationConfig, RandomSizedCropConfig,
    ResizeConfig, RotateConfig, RotationAngle, SafeRotateConfig, SquareSymmetry, TenCropConfig,
    ThinPlateSplineConfig, WaterRefractionConfig,
};
pub(crate) use spatial::{
    homography, pillow_affine_map, quarter_turn, rotation_inverse, sample_border_crop, sample_scale,
};

mod rgb;
pub(crate) use rgb::{
    contiguous_values, from_rgb_image, into_rgb_image, logical_offset, require_u8_hwc,
    with_rgb_pixels,
};

mod v2;
pub use degradation::{LensFlareConfig, RandomSunFlareConfig};
pub use filtering::AdvancedBlurConfig;
pub use photometric::{IlluminationEffect, IlluminationMode, RandomShadowConfig};
pub use v2::V2Config;
