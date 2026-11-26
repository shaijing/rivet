//! Configuration and dispatch for image operators behind the v2 frontend.

mod execute;
mod metadata;
#[cfg(test)]
mod tests;
mod validate;

use crate::transforms::{
    AdditiveNoiseConfig, AdvancedBlurConfig, AnnotationArtifactsConfig, AtmosphericFogConfig,
    ChannelDropoutConfig, ColorizeConfig, CropAndPadConfig, DitheringConfig, EnhanceMode,
    FdaConfig, GridDistortionConfig, GridElasticDeformConfig, GridMaskConfig, HEStainConfig,
    HistogramMatchingConfig, IlluminationEffect, IlluminationMode, InterpolationMode,
    LensFlareConfig, LetterBoxConfig, LongestMaxSizeConfig, MorphologicalConfig,
    OpticalDistortionConfig, PadIfNeededConfig, PaddingMode, PiecewiseAffineConfig,
    PixelDistributionAdaptationConfig, PixelDropoutConfig, PixelSpreadConfig, RandomFogConfig,
    RandomGravelConfig, RandomGridShuffleConfig, RandomRainConfig, RandomShadowConfig,
    RandomSizedCropConfig, RandomSnowConfig, RandomSunFlareConfig, SafeRotateConfig, SpatterConfig,
    SquareSymmetry, SuperpixelsConfig, ThinPlateSplineConfig, WaterRefractionConfig,
    XyMaskingConfig,
};
use rivet_core::DType;

#[derive(Clone, Debug, PartialEq)]
pub enum V2Config {
    PermuteChannels(Vec<usize>),
    RandomChannelPermutation {
        probability: f64,
    },
    RandomPhotometricDistort {
        brightness: (f32, f32),
        contrast: (f32, f32),
        saturation: (f32, f32),
        hue: (f32, f32),
        probability: f64,
    },
    RandomCropFromBorders {
        fractions: [f64; 4],
    },
    RandomResize {
        min_size: u32,
        max_size: u32,
        interpolation: InterpolationMode,
    },
    RandomShortestSize {
        min_sizes: Vec<u32>,
        max_size: Option<u32>,
        interpolation: InterpolationMode,
    },
    ScaleJitter {
        target_width: u32,
        target_height: u32,
        scale_range: (f64, f64),
        interpolation: InterpolationMode,
    },
    RandomZoomOut {
        fill: Vec<f32>,
        side_range: (f64, f64),
        probability: f64,
    },
    GaussianNoise {
        mean: f64,
        sigma: f64,
        clip: bool,
    },
    BiasField(crate::volume::BiasFieldConfig),
    GibbsRinging(crate::volume::GibbsRingingConfig),
    HEStain(HEStainConfig),
    KSpaceSpikeNoise(crate::volume::KSpaceSpikeNoiseConfig),
    RicianNoise {
        std_range: [f64; 2],
        per_channel: bool,
    },
    ShotNoise {
        intensity: f32,
    },
    RandomToneCurve {
        scale: f64,
    },
    Clahe {
        grid: (u32, u32),
        clip_limit: f64,
    },
    AdditiveNoise(std::sync::Arc<AdditiveNoiseConfig>),
    MultiplicativeNoise {
        multiplier: (f64, f64),
        per_channel: bool,
        elementwise: bool,
    },
    RandomSunFlare(std::sync::Arc<RandomSunFlareConfig>),
    HueSaturationValue(std::sync::Arc<[(f64, f64); 3]>),
    Halftone {
        dot_size_range: (u32, u32),
        blend_range: (f64, f64),
    },
    Dithering(DitheringConfig),
    RingingOvershoot {
        blur_range: (u32, u32),
        cutoff_range: (f64, f64),
    },
    Enhance {
        mode: EnhanceMode,
        alpha_range: (f64, f64),
    },
    Emboss {
        alpha_range: (f64, f64),
        strength_range: (f64, f64),
    },
    ExposureMatching {
        target_mean_range: (f64, f64),
        gain_range: Option<(f64, f64)>,
    },
    FancyPca {
        alpha: f64,
    },
    Colorize(ColorizeConfig),
    ToSepia {
        alpha: f64,
    },
    RgbShift(std::sync::Arc<[(f64, f64); 3]>),
    RandomBrightnessContrast {
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        brightness_by_max: bool,
        ensure_safe_output: bool,
    },
    RandomGamma {
        gamma_range: (f64, f64),
    },
    RandomFog(RandomFogConfig),
    AtmosphericFog(AtmosphericFogConfig),
    Spatter(std::sync::Arc<SpatterConfig>),
    RandomGravel(RandomGravelConfig),
    RandomSnow(RandomSnowConfig),
    RandomRain(std::sync::Arc<RandomRainConfig>),
    LensFlare(std::sync::Arc<LensFlareConfig>),
    RandomShadow(std::sync::Arc<RandomShadowConfig>),
    PlasmaShadow {
        shadow_intensity_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    },
    PlasmaBrightnessContrast {
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    },
    Illumination {
        intensity_range: (f64, f64),
        mode: IlluminationMode,
        effect: IlluminationEffect,
    },
    FilmGrain {
        intensity_range: (f64, f64),
        grain_size_range: (u32, u32),
        per_channel: bool,
    },
    IsoNoise {
        color_shift_range: (f64, f64),
        intensity_range: (f64, f64),
    },
    GlassBlur {
        sigma: f64,
        max_delta: u32,
        iterations: u32,
    },
    ZoomBlur {
        max_factor_range: (f64, f64),
        step_factor_range: (f64, f64),
    },
    StochasticConvolution {
        kernel_range: (u32, u32),
        strength_range: (f32, f32),
        per_channel: bool,
        padding_mode: PaddingMode,
    },
    AdvancedBlur(std::sync::Arc<AdvancedBlurConfig>),
    Blur {
        kernel_range: (u32, u32),
    },
    MedianBlur,
    PlanckianJitter {
        temperature_range: (f64, f64),
    },
    Vignetting {
        intensity_range: (f64, f64),
        center_range: (f64, f64),
    },
    ChromaticAberration {
        red_range: (f64, f64),
        blue_range: (f64, f64),
        interpolation: InterpolationMode,
    },
    PadIfNeeded(std::sync::Arc<PadIfNeededConfig>),
    LetterBox(LetterBoxConfig),
    Defocus {
        radius_range: (u32, u32),
    },
    ChannelDropout(std::sync::Arc<ChannelDropoutConfig>),
    XyMasking(std::sync::Arc<XyMaskingConfig>),
    PixelDropout(std::sync::Arc<PixelDropoutConfig>),
    CoarseDropout {
        holes: (u32, u32),
        hole_height: (u32, u32),
        hole_width: (u32, u32),
        fill: f32,
    },
    MotionBlur {
        length_range: (u32, u32),
        angle_range: (f64, f64),
    },
    UnsharpMask {
        sigma: f32,
        amount: f32,
        threshold: u8,
    },
    Downscale {
        scale_range: (f64, f64),
        down_interpolation: InterpolationMode,
        up_interpolation: InterpolationMode,
    },
    SaltAndPepper {
        amount: f64,
        salt_ratio: f64,
    },
    GridDropout {
        unit_size: usize,
        ratio: f64,
        fill: f32,
    },
    RandomScale {
        range: (f64, f64),
        y: Option<(f64, f64)>,
        interpolation: InterpolationMode,
    },
    LongestMaxSize(LongestMaxSizeConfig),
    Jpeg {
        quality: (u8, u8),
    },
    Rgb,
    ToDtype {
        dtype: DType,
        scale: bool,
    },
    SquareSymmetry(Option<SquareSymmetry>),
    AnnotationArtifacts(std::sync::Arc<AnnotationArtifactsConfig>),
    Fda(std::sync::Arc<FdaConfig>),
    PixelDistributionAdaptation(std::sync::Arc<PixelDistributionAdaptationConfig>),
    HistogramMatching(std::sync::Arc<HistogramMatchingConfig>),
    WaterRefraction(WaterRefractionConfig),
    RandomGridShuffle(RandomGridShuffleConfig),
    PixelSpread(PixelSpreadConfig),
    ThinPlateSpline(ThinPlateSplineConfig),
    GridElasticDeform(GridElasticDeformConfig),
    PiecewiseAffine(PiecewiseAffineConfig),
    GridDistortion(GridDistortionConfig),
    OpticalDistortion(OpticalDistortionConfig),
    Superpixels(SuperpixelsConfig),
    Morphological(MorphologicalConfig),
    SafeRotate(SafeRotateConfig),
    RandomRotate90,
    Transpose,
    GridMask(std::sync::Arc<GridMaskConfig>),
    RandomSizedCrop(RandomSizedCropConfig),
    CropAndPad(std::sync::Arc<CropAndPadConfig>),
    Webp {
        quality: (u8, u8),
    },
}
