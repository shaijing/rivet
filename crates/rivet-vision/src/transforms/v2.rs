use super::*;
use crate::sample::image::{ImageAxisOrder, ImageSample};
use rivet_data::random::RandomStream;
#[derive(Clone, Debug, PartialEq)]
pub struct AdvancedBlurConfig {
    pub kernel_range: (u32, u32),
    pub sigma_x_range: (f64, f64),
    pub sigma_y_range: (f64, f64),
    pub rotate_range: (f64, f64),
    pub beta_range: (f64, f64),
    pub noise_range: (f64, f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IlluminationMode {
    Linear,
    Corner,
    Gaussian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IlluminationEffect {
    Brighten,
    Darken,
    Both,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomShadowConfig {
    pub shadow_roi: (f64, f64, f64, f64),
    pub num_shadows_range: (u32, u32),
    pub shadow_dimension: u32,
    pub shadow_intensity_range: (f64, f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomSunFlareConfig {
    pub flare_roi: (f64, f64, f64, f64),
    pub angle_range: (f64, f64),
    pub num_flare_circles_range: (u32, u32),
    pub src_radius: u32,
    pub src_color: (u8, u8, u8),
    pub physics_based: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LensFlareConfig {
    pub flare_roi: (f64, f64, f64, f64),
    pub num_ghosts_range: (u32, u32),
    pub intensity_range: (f64, f64),
    pub num_rays_range: (u32, u32),
    pub bloom_range: (f64, f64),
}

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
}
impl V2Config {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PermuteChannels(_) => "PermuteChannels",
            Self::RandomChannelPermutation { .. } => "RandomChannelPermutation",
            Self::RandomPhotometricDistort { .. } => "RandomPhotometricDistort",
            Self::RandomCropFromBorders { .. } => "RandomCropFromBorders",
            Self::RandomSizedCrop(_) => "RandomSizedCrop",
            Self::RandomResize { .. } => "RandomResize",
            Self::RandomShortestSize { .. } => "RandomShortestSize",
            Self::ScaleJitter { .. } => "ScaleJitter",
            Self::RandomZoomOut { .. } => "RandomZoomOut",
            Self::GaussianNoise { .. } => "GaussianNoise",
            Self::BiasField(_) => "BiasField",
            Self::GibbsRinging(_) => "GibbsRinging",
            Self::RicianNoise { .. } => "RicianNoise",
            Self::ShotNoise { .. } => "ShotNoise",
            Self::RandomToneCurve { .. } => "RandomToneCurve",
            Self::Clahe { .. } => "CLAHE",
            Self::AdditiveNoise(_) => "AdditiveNoise",
            Self::MultiplicativeNoise { .. } => "MultiplicativeNoise",
            Self::RandomSunFlare(_) => "RandomSunFlare",
            Self::HueSaturationValue(_) => "HueSaturationValue",
            Self::Halftone { .. } => "Halftone",
            Self::Dithering(_) => "Dithering",
            Self::RingingOvershoot { .. } => "RingingOvershoot",
            Self::Enhance { .. } => "Enhance",
            Self::Emboss { .. } => "Emboss",
            Self::ExposureMatching { .. } => "ExposureMatching",
            Self::FancyPca { .. } => "FancyPCA",
            Self::Colorize(_) => "Colorize",
            Self::ToSepia { .. } => "ToSepia",
            Self::RgbShift(_) => "RGBShift",
            Self::RandomBrightnessContrast { .. } => "RandomBrightnessContrast",
            Self::RandomGamma { .. } => "RandomGamma",
            Self::RandomFog(_) => "RandomFog",
            Self::AtmosphericFog(_) => "AtmosphericFog",
            Self::Spatter(_) => "Spatter",
            Self::RandomGravel(_) => "RandomGravel",
            Self::RandomSnow(_) => "RandomSnow",
            Self::RandomRain(_) => "RandomRain",
            Self::LensFlare(_) => "LensFlare",
            Self::RandomShadow(_) => "RandomShadow",
            Self::PlasmaShadow { .. } => "PlasmaShadow",
            Self::PlasmaBrightnessContrast { .. } => "PlasmaBrightnessContrast",
            Self::Illumination { .. } => "Illumination",
            Self::FilmGrain { .. } => "FilmGrain",
            Self::IsoNoise { .. } => "ISONoise",
            Self::GlassBlur { .. } => "GlassBlur",
            Self::ZoomBlur { .. } => "ZoomBlur",
            Self::StochasticConvolution { .. } => "StochasticConvolution",
            Self::AdvancedBlur(_) => "AdvancedBlur",
            Self::Blur { .. } => "Blur",
            Self::MedianBlur => "MedianBlur",
            Self::PlanckianJitter { .. } => "PlanckianJitter",
            Self::Vignetting { .. } => "Vignetting",
            Self::ChromaticAberration { .. } => "ChromaticAberration",
            Self::PadIfNeeded(_) => "PadIfNeeded",
            Self::LetterBox(_) => "LetterBox",
            Self::Defocus { .. } => "Defocus",
            Self::ChannelDropout(_) => "ChannelDropout",
            Self::SquareSymmetry(_) => "SquareSymmetry",
            Self::Fda(_) => "FDA",
            Self::PixelDistributionAdaptation(_) => "PixelDistributionAdaptation",
            Self::HistogramMatching(_) => "HistogramMatching",
            Self::WaterRefraction(_) => "WaterRefraction",
            Self::RandomGridShuffle(_) => "RandomGridShuffle",
            Self::PixelSpread(_) => "PixelSpread",
            Self::ThinPlateSpline(_) => "ThinPlateSpline",
            Self::GridElasticDeform(_) => "GridElasticDeform",
            Self::PiecewiseAffine(_) => "PiecewiseAffine",
            Self::GridDistortion(_) => "GridDistortion",
            Self::OpticalDistortion(_) => "OpticalDistortion",
            Self::Superpixels(_) => "Superpixels",
            Self::Morphological(_) => "Morphological",
            Self::SafeRotate(_) => "SafeRotate",
            Self::RandomRotate90 => "RandomRotate90",
            Self::Transpose => "Transpose",
            Self::GridMask(_) => "GridMask",
            Self::XyMasking(_) => "XYMasking",
            Self::PixelDropout(_) => "PixelDropout",
            Self::CoarseDropout { .. } => "CoarseDropout",
            Self::MotionBlur { .. } => "MotionBlur",
            Self::UnsharpMask { .. } => "UnsharpMask",
            Self::Downscale { .. } => "Downscale",
            Self::SaltAndPepper { .. } => "SaltAndPepper",
            Self::GridDropout { .. } => "GridDropout",
            Self::RandomScale { .. } => "RandomScale",
            Self::LongestMaxSize(_) => "LongestMaxSize",
            Self::CropAndPad(_) => "CropAndPad",
            Self::Jpeg { .. } => "JPEG",
            Self::Rgb => "RGB",
            Self::ToDtype { .. } => "ToDtype",
        }
    }
    pub const fn planning_op_id(&self) -> u32 {
        match self {
            Self::PermuteChannels(_) => 49,
            Self::RandomChannelPermutation { .. } => 50,
            Self::RandomPhotometricDistort { .. } => 51,
            Self::RandomCropFromBorders { .. } => 114,
            Self::RandomSizedCrop(_) => 115,
            Self::RandomResize { .. } => 52,
            Self::RandomShortestSize { .. } => 53,
            Self::ScaleJitter { .. } => 54,
            Self::RandomZoomOut { .. } => 55,
            Self::GaussianNoise { .. } => 56,
            Self::BiasField(_) => 139,
            Self::GibbsRinging(_) => 140,
            Self::RicianNoise { .. } => 138,
            Self::ShotNoise { .. } => 60,
            Self::RandomToneCurve { .. } => 69,
            Self::Clahe { .. } => 70,
            Self::AdditiveNoise(_) => 83,
            Self::MultiplicativeNoise { .. } => 82,
            Self::RandomSunFlare(_) => 88,
            Self::HueSaturationValue(_) => 99,
            Self::Halftone { .. } => 108,
            Self::Dithering(_) => 107,
            Self::RingingOvershoot { .. } => 106,
            Self::Enhance { .. } => 105,
            Self::Emboss { .. } => 104,
            Self::ExposureMatching { .. } => 103,
            Self::FancyPca { .. } => 102,
            Self::Colorize(_) => 101,
            Self::ToSepia { .. } => 100,
            Self::RgbShift(_) => 97,
            Self::RandomBrightnessContrast { .. } => 98,
            Self::RandomGamma { .. } => 96,
            Self::RandomFog(_) => 94,
            Self::AtmosphericFog(_) => 95,
            Self::Spatter(_) => 93,
            Self::RandomGravel(_) => 92,
            Self::RandomSnow(_) => 91,
            Self::RandomRain(_) => 90,
            Self::LensFlare(_) => 89,
            Self::RandomShadow(_) => 87,
            Self::PlasmaShadow { .. } => 85,
            Self::PlasmaBrightnessContrast { .. } => 86,
            Self::Illumination { .. } => 84,
            Self::FilmGrain { .. } => 81,
            Self::IsoNoise { .. } => 80,
            Self::GlassBlur { .. } => 79,
            Self::ZoomBlur { .. } => 78,
            Self::StochasticConvolution { .. } => 77,
            Self::AdvancedBlur(_) => 76,
            Self::Blur { .. } => 75,
            Self::MedianBlur => 74,
            Self::PlanckianJitter { .. } => 73,
            Self::Vignetting { .. } => 72,
            Self::ChromaticAberration { .. } => 71,
            Self::PadIfNeeded(_) => 117,
            Self::LetterBox(_) => 66,
            Self::Defocus { .. } => 67,
            Self::ChannelDropout(_) => 111,
            Self::SquareSymmetry(_) => 122,
            Self::Fda(_) => 137,
            Self::PixelDistributionAdaptation(_) => 136,
            Self::HistogramMatching(_) => 135,
            Self::WaterRefraction(_) => 133,
            Self::RandomGridShuffle(_) => 132,
            Self::PixelSpread(_) => 131,
            Self::ThinPlateSpline(_) => 130,
            Self::GridElasticDeform(_) => 129,
            Self::PiecewiseAffine(_) => 128,
            Self::GridDistortion(_) => 127,
            Self::OpticalDistortion(_) => 126,
            Self::Superpixels(_) => 125,
            Self::Morphological(_) => 124,
            Self::SafeRotate(_) => 123,
            Self::RandomRotate90 => 121,
            Self::Transpose => 120,
            Self::GridMask(_) => 113,
            Self::XyMasking(_) => 112,
            Self::PixelDropout(_) => 110,
            Self::CoarseDropout { .. } => 68,
            Self::MotionBlur { .. } => 64,
            Self::UnsharpMask { .. } => 65,
            Self::Downscale { .. } => 61,
            Self::SaltAndPepper { .. } => 62,
            Self::GridDropout { .. } => 63,
            Self::RandomScale { .. } => 119,
            Self::LongestMaxSize(_) => 118,
            Self::CropAndPad(_) => 116,
            Self::Jpeg { .. } => 57,
            Self::Rgb => 58,
            Self::ToDtype { .. } => 59,
        }
    }
    pub fn is_random(&self) -> bool {
        if let Self::LongestMaxSize(config) = self {
            return config.random();
        }
        if let Self::PadIfNeeded(config) = self {
            return config.position == PadPosition::Random;
        }
        if let Self::Dithering(config) = self {
            return matches!(config.method, DitherMethod::Random { .. });
        }
        !matches!(
            self,
            Self::PermuteChannels(_)
                | Self::Rgb
                | Self::ToDtype { .. }
                | Self::UnsharpMask { .. }
                | Self::LetterBox(_)
                | Self::ToSepia { .. }
                | Self::SquareSymmetry(Some(_))
                | Self::Morphological(_)
                | Self::Transpose
                | Self::MedianBlur
                | Self::Clahe { .. }
        )
    }
    pub fn validate(&self) -> RivetResult<()> {
        let valid = match self {
            Self::Fda(config) => config.valid(),
            Self::PixelDistributionAdaptation(config) => config.valid(),
            Self::HistogramMatching(config) => config.valid(),
            Self::WaterRefraction(config) => config.valid(),
            Self::RandomGridShuffle(config) => config.valid(),
            Self::PixelSpread(config) => config.valid(),
            Self::ThinPlateSpline(config) => config.valid(),
            Self::GridElasticDeform(config) => config.valid(),
            Self::PiecewiseAffine(config) => config.valid(),
            Self::GridDistortion(config) => config.valid(),
            Self::OpticalDistortion(config) => config.valid(),
            Self::Superpixels(config) => config.valid(),
            Self::Morphological(config) => config.valid(),
            Self::SafeRotate(config) => return config.validate(),
            Self::RandomScale { range, y, .. } => {
                let valid = |(a, b): (f64, f64)| a > -1.0 && a <= b && b.is_finite();
                valid(*range) && y.is_none_or(valid)
            }

            Self::LongestMaxSize(config) => config.valid(),
            Self::PadIfNeeded(config) => config.valid(),
            Self::GridMask(config) => config.valid(),
            Self::RandomCropFromBorders { fractions } => {
                fractions.iter().all(|v| (0.0..=1.0).contains(v))
            }
            Self::RandomSizedCrop(config) => config.valid(),
            Self::CropAndPad(config) => config.valid(),
            Self::XyMasking(config) => config.valid(),
            Self::ChannelDropout(config) => config.valid(),
            Self::PixelDropout(config) => config.valid(),
            Self::Halftone {
                dot_size_range: (lo, hi),
                blend_range: (min, max),
            } => *lo >= 2 && lo <= hi && *min >= 0.0 && min <= max && *max <= 1.0,
            Self::Dithering(config) => config.valid(),
            Self::RingingOvershoot {
                blur_range: (lo, hi),
                cutoff_range: (min, max),
            } => {
                *lo >= 3
                    && lo <= hi
                    && *hi <= 127
                    && lo % 2 == 1
                    && hi % 2 == 1
                    && *min > 0.0
                    && min <= max
                    && *max < std::f64::consts::PI
            }
            Self::Enhance {
                alpha_range: (lo, hi),
                ..
            } => *lo >= 0.0 && lo <= hi && *hi <= f64::from(f32::MAX),
            Self::Emboss {
                alpha_range: (lo, hi),
                strength_range: (min, max),
            } => {
                *lo >= 0.0
                    && lo <= hi
                    && *hi <= 1.0
                    && *min >= 0.0
                    && min <= max
                    && *max <= f64::from(f32::MAX)
            }
            Self::ExposureMatching {
                target_mean_range: (lo, hi),
                gain_range,
            } => {
                *lo >= 0.0
                    && lo <= hi
                    && *hi <= 1.0
                    && gain_range.is_none_or(|(lo, hi)| lo >= 0.0 && lo <= hi && hi.is_finite())
            }
            Self::FancyPca { alpha } => alpha.is_finite() && *alpha >= 0.0,
            Self::Colorize(config) => config.valid(),
            Self::ToSepia { alpha } => (0.0..=1.0).contains(alpha),
            Self::HueSaturationValue(ranges) => ranges
                .iter()
                .zip([180.0, 255.0, 255.0])
                .all(|(&(lo, hi), limit)| lo >= -limit && lo <= hi && hi <= limit),
            Self::RandomBrightnessContrast {
                brightness_range,
                contrast_range,
                ..
            } => [brightness_range, contrast_range]
                .into_iter()
                .all(|&(lo, hi)| lo >= -1.0 && lo <= hi && hi <= f64::from(f32::MAX)),
            Self::RgbShift(ranges) => ranges
                .iter()
                .all(|&(lo, hi)| lo >= -255.0 && lo <= hi && hi <= 255.0),
            Self::RandomGamma {
                gamma_range: (lo, hi),
            } => *lo > 0.0 && lo <= hi && hi.is_finite(),
            Self::RandomFog(config) => config.valid(),
            Self::AtmosphericFog(config) => config.valid(),
            Self::Spatter(config) => config.valid(),
            Self::RandomGravel(config) => config.valid(),
            Self::RandomSnow(config) => config.valid(),
            Self::RandomRain(config) => config.valid(),
            Self::LensFlare(config) => super::flare::lens_flare_valid(config),
            Self::RandomSunFlare(config) => super::flare::random_sun_flare_valid(config),
            Self::RandomShadow(config) => {
                let (x0, y0, x1, y1) = config.shadow_roi;
                let (min, max) = config.num_shadows_range;
                let (low, high) = config.shadow_intensity_range;
                x0 >= 0.0
                    && x0 < x1
                    && x1 <= 1.0
                    && y0 >= 0.0
                    && y0 < y1
                    && y1 <= 1.0
                    && min <= max
                    && max <= 64
                    && (3..=64).contains(&config.shadow_dimension)
                    && low >= 0.0
                    && low <= high
                    && high <= 1.0
            }
            Self::PlasmaBrightnessContrast {
                brightness_range,
                contrast_range,
                plasma_size,
                roughness,
            } => {
                brightness_range.0 >= -1.0
                    && brightness_range.0 <= brightness_range.1
                    && brightness_range.1 <= 1.0
                    && contrast_range.0 >= -1.0
                    && contrast_range.0 <= contrast_range.1
                    && contrast_range.1 <= 1.0
                    && (2..=1024).contains(plasma_size)
                    && roughness.is_finite()
                    && *roughness > 0.0
            }
            Self::PlasmaShadow {
                shadow_intensity_range,
                plasma_size,
                roughness,
            } => {
                shadow_intensity_range.0 >= 0.0
                    && shadow_intensity_range.0 <= shadow_intensity_range.1
                    && shadow_intensity_range.1 <= 1.0
                    && (2..=1024).contains(plasma_size)
                    && roughness.is_finite()
                    && *roughness > 0.0
            }
            Self::Illumination {
                intensity_range: (low, high),
                ..
            } => *low >= 0.0 && low <= high && *high <= 1.0,
            Self::AdditiveNoise(config) => config.distribution.valid(config.sharing),
            Self::MultiplicativeNoise {
                multiplier: (low, high),
                ..
            } => *low >= 0.0 && low <= high && high.is_finite(),
            Self::FilmGrain {
                intensity_range: (low, high),
                grain_size_range: (min, max),
                ..
            } => *low >= 0.0 && low <= high && *high <= 1.0 && *min > 0 && min <= max,
            Self::IsoNoise {
                color_shift_range,
                intensity_range,
            } => [color_shift_range, intensity_range]
                .into_iter()
                .all(|&(low, high)| low >= 0.0 && low <= high && high <= 1.0),
            Self::GlassBlur {
                sigma,
                max_delta,
                iterations,
            } => (0.0..=10.0).contains(sigma) && *max_delta > 0 && (1..=32).contains(iterations),
            Self::ZoomBlur {
                max_factor_range: (low, high),
                step_factor_range: (step_low, step_high),
            } => {
                *low >= 1.0
                    && low <= high
                    && high.is_finite()
                    && *step_low > 0.0
                    && step_low <= step_high
                    && step_high.is_finite()
                    && (high - 1.0) / step_low <= 128.0
            }
            Self::StochasticConvolution {
                kernel_range,
                strength_range: (low, high),
                ..
            } => {
                Self::Blur {
                    kernel_range: *kernel_range,
                }
                .validate()
                .is_ok()
                    && *low >= 0.0
                    && low <= high
                    && high.is_finite()
            }
            Self::AdvancedBlur(config) => {
                let AdvancedBlurConfig {
                    kernel_range,
                    sigma_x_range,
                    sigma_y_range,
                    rotate_range,
                    beta_range,
                    noise_range,
                } = config.as_ref();
                let positive =
                    |&(low, high): &(f64, f64)| low > 0.0 && low <= high && high.is_finite();
                Self::Blur {
                    kernel_range: *kernel_range,
                }
                .validate()
                .is_ok()
                    && [sigma_x_range, sigma_y_range, beta_range, noise_range]
                        .into_iter()
                        .all(positive)
                    && rotate_range.0 >= -360.0
                    && rotate_range.0 <= rotate_range.1
                    && rotate_range.1 <= 360.0
            }
            Self::Blur {
                kernel_range: (low, high),
            } => *low >= 3 && low <= high && *high <= 127 && low % 2 == 1 && high % 2 == 1,
            Self::MedianBlur => true,
            Self::RandomChannelPermutation { probability } => {
                probability.is_finite() && (0.0..=1.0).contains(probability)
            }
            Self::RandomPhotometricDistort {
                brightness,
                contrast,
                saturation,
                hue,
                probability,
            } => {
                probability.is_finite()
                    && (0.0..=1.0).contains(probability)
                    && [brightness, contrast, saturation]
                        .iter()
                        .all(|(a, b)| a.is_finite() && b.is_finite() && *a >= 0.0 && b >= a)
                    && hue.0.is_finite()
                    && hue.1.is_finite()
                    && hue.0 >= -0.5
                    && hue.1 <= 0.5
                    && hue.1 >= hue.0
            }
            Self::RandomResize {
                min_size, max_size, ..
            } => *min_size > 0 && max_size > min_size,
            Self::RandomShortestSize {
                min_sizes,
                max_size,
                ..
            } => {
                !min_sizes.is_empty()
                    && min_sizes.iter().all(|x| *x > 0)
                    && max_size.is_none_or(|m| m > 0)
            }
            Self::ScaleJitter {
                target_width,
                target_height,
                scale_range: (a, b),
                ..
            } => {
                *target_width > 0
                    && *target_height > 0
                    && a.is_finite()
                    && b.is_finite()
                    && *a > 0.0
                    && b >= a
            }
            Self::RandomZoomOut {
                side_range: (a, b),
                probability,
                fill,
            } => {
                a.is_finite()
                    && b.is_finite()
                    && *a >= 1.0
                    && b >= a
                    && probability.is_finite()
                    && (0.0..=1.0).contains(probability)
                    && matches!(fill.len(), 1 | 3)
                    && fill.iter().all(|x| x.is_finite())
            }
            Self::GaussianNoise { mean, sigma, .. } => {
                return super::noise::validate_gaussian_noise(*mean, *sigma);
            }
            Self::BiasField(config) => return config.validate(),
            Self::GibbsRinging(config) => return config.validate(),
            Self::RicianNoise { std_range, .. } => {
                return super::noise::validate_rician_std(*std_range);
            }
            Self::ShotNoise { intensity } => return super::noise::validate_shot_noise(*intensity),
            Self::PlanckianJitter {
                temperature_range: (low, high),
            } => *low >= 3000.0 && low <= high && *high <= 15000.0,
            Self::Vignetting {
                intensity_range,
                center_range,
            } => [intensity_range, center_range]
                .into_iter()
                .all(|&(low, high)| low >= 0.0 && low <= high && high <= 1.0),
            Self::RandomToneCurve { scale } => scale.is_finite() && *scale >= 0.0,
            Self::Clahe { grid, clip_limit } => {
                grid.0 > 0 && grid.1 > 0 && clip_limit.is_finite() && *clip_limit > 0.0
            }
            Self::ChromaticAberration {
                red_range,
                blue_range,
                interpolation,
            } => {
                [red_range, blue_range]
                    .into_iter()
                    .all(|&(low, high)| low >= -1.0 && low <= high && high <= 1.0)
                    && matches!(
                        interpolation,
                        InterpolationMode::Nearest | InterpolationMode::Bilinear
                    )
            }
            Self::LetterBox(config) => return config.validate(),
            Self::Defocus {
                radius_range: (low, high),
            } => low <= high && *high <= 127,
            Self::CoarseDropout {
                holes,
                hole_height,
                hole_width,
                fill,
            } => {
                holes.0 <= holes.1
                    && hole_height.0 > 0
                    && hole_width.0 > 0
                    && hole_height.0 <= hole_height.1
                    && hole_width.0 <= hole_width.1
                    && fill.is_finite()
            }
            Self::MotionBlur {
                length_range: (low, high),
                angle_range: (start, end),
            } => {
                *low > 0
                    && high >= low
                    && low % 2 == 1
                    && high % 2 == 1
                    && *high <= 255
                    && start.is_finite()
                    && end.is_finite()
                    && end >= start
                    && (end - start).is_finite()
            }
            Self::UnsharpMask { sigma, amount, .. } => {
                sigma.is_finite() && *sigma > 0.0 && amount.is_finite() && *amount >= 0.0
            }
            Self::Downscale {
                scale_range: (a, b),
                ..
            } => *a > 0.0 && b >= a && *b <= 1.0,
            Self::SaltAndPepper { amount, salt_ratio } => {
                (0.0..=1.0).contains(amount) && (0.0..=1.0).contains(salt_ratio)
            }
            Self::GridDropout {
                unit_size,
                ratio,
                fill,
            } => *unit_size >= 2 && *ratio > 0.0 && *ratio < 1.0 && fill.is_finite(),
            Self::Jpeg { quality: (a, b) } => *a >= 1 && b >= a && *b <= 100,
            Self::ToDtype { dtype, .. } => matches!(dtype, DType::U8 | DType::F32),
            _ => true,
        };
        if valid {
            Ok(())
        } else {
            Err(invalid_argument(format!(
                "invalid {} parameters",
                self.name()
            )))
        }
    }
    pub fn apply_batch(&self, input: Tensor, axis: ImageAxisOrder) -> RivetResult<Tensor> {
        self.validate()?;
        if input.rank() != 4 {
            return Err(invalid_shape("batch ToDtype requires rank-4 image input"));
        }
        if let Self::ToDtype { dtype, scale } = self {
            if *scale {
                ConvertImageDtypeConfig::new(*dtype).apply_batch(input, axis)
            } else {
                super::v2_numeric::cast_unscaled(&input, *dtype)
            }
        } else {
            Err(invalid_argument("only ToDtype supports batches"))
        }
    }
    pub fn apply(
        &self,
        sample: ImageSample,
        axis: ImageAxisOrder,
        rng: &mut RandomStream,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        if let Self::ToDtype { dtype, scale } = self {
            return if *scale {
                ConvertImageDtypeConfig::new(*dtype).apply(sample, axis)
            } else {
                let mut s = sample.into_decoded()?;
                s.image = super::v2_numeric::cast_unscaled(&s.image, *dtype)?;
                Ok(ImageSample::Decoded(s))
            };
        }
        let mut s = sample.into_decoded()?;
        let dims = s.image.dims().to_vec();
        if dims.len() != 3 {
            return Err(invalid_argument("v2 image requires rank 3"));
        }
        let (h, w, c) = match axis {
            ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
            ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
        };
        match self {
            Self::RandomPhotometricDistort {
                brightness,
                contrast,
                saturation,
                hue,
                probability,
            } => {
                let factor =
                    |range: (f32, f32), rng: &mut RandomStream| -> RivetResult<Option<f32>> {
                        Ok(if rng.gen_bool(*probability)? {
                            Some(range.0 + (range.1 - range.0) * rng.next_f64() as f32)
                        } else {
                            None
                        })
                    };
                let b = factor(*brightness, rng)?;
                let ct = factor(*contrast, rng)?;
                let sat = factor(*saturation, rng)?;
                let hu = factor(*hue, rng)?;
                let before = rng.gen_bool(0.5)?;
                let permute_channels = rng.gen_bool(*probability)?;
                let mut order: Vec<usize> = (0..c).collect();
                if permute_channels {
                    for i in (1..c).rev() {
                        let j = rng.gen_range_usize(0..i + 1)?;
                        order.swap(i, j);
                    }
                }
                let mut sample = ImageSample::Decoded(s);
                if let Some(v) = b {
                    sample = BrightnessConfig::new(v).apply(sample)?;
                }
                if before {
                    if let Some(v) = ct {
                        sample = ContrastConfig::new(v).apply(sample)?;
                    }
                }
                if let Some(v) = sat {
                    sample = SaturationConfig::new(v).apply(sample)?;
                }
                if let Some(v) = hu {
                    sample = HueConfig::new(v).apply(sample)?;
                }
                if !before {
                    if let Some(v) = ct {
                        sample = ContrastConfig::new(v).apply(sample)?;
                    }
                }
                if permute_channels {
                    let mut out = sample.into_decoded()?;
                    out.image = permute(&out.image, &dims, axis, &order)?;
                    sample = ImageSample::Decoded(out);
                }
                return Ok(sample);
            }
            Self::RandomZoomOut {
                fill,
                side_range: (a, b),
                probability,
            } => {
                if !rng.gen_bool(*probability)? {
                    return Ok(ImageSample::Decoded(s));
                }
                let scale = a + rng.next_f64() * (b - a);
                let ow = (w as f64 * scale).floor();
                let oh = (h as f64 * scale).floor();
                if ow > u32::MAX as f64 || oh > u32::MAX as f64 {
                    return Err(invalid_argument("zoom out dimensions exceed u32"));
                }
                let dx = ow as u32 - w as u32;
                let dy = oh as u32 - h as u32;
                let left = (dx as f64 * rng.next_f64()) as u32;
                let top = (dy as f64 * rng.next_f64()) as u32;
                let mut config = PadConfig::with_sides(left, top, dx - left, dy - top);
                config = if fill.len() == 1 {
                    config.with_fill(fill[0])
                } else {
                    config.with_fill_channels([fill[0], fill[1], fill[2]])
                };
                return config.apply(ImageSample::Decoded(s), axis);
            }
            Self::RandomCropFromBorders { fractions } => {
                let [x, y, w, h] = super::sample_border_crop(h, w, *fractions, rng)?;
                return CropConfig::new(x, y, w, h).apply(ImageSample::Decoded(s), axis);
            }
            Self::RandomSizedCrop(config) => {
                s = require_u8_hwc(s, self.name())?;
                let [x, y, w, h] = config.sample(h, w, rng)?;
                let cropped = CropConfig::new(x, y, w, h).apply(ImageSample::Decoded(s), axis)?;
                return ResizeConfig::with_interpolation(
                    config.width,
                    config.height,
                    config.interpolation,
                )
                .apply(cropped);
            }
            Self::RandomResize {
                min_size,
                max_size,
                interpolation,
            } => {
                return ResizeConfig::short_edge(
                    rng.gen_range_usize(*min_size as usize..*max_size as usize)? as u32,
                    None,
                    *interpolation,
                )
                .apply(ImageSample::Decoded(s));
            }
            Self::RandomShortestSize {
                min_sizes,
                max_size,
                interpolation,
            } => {
                let short = min_sizes[rng.gen_range_usize(0..min_sizes.len())?] as f64;
                let mut ratio = short / (h.min(w) as f64);
                if let Some(max) = max_size {
                    ratio = ratio.min(*max as f64 / (h.max(w) as f64));
                }
                let ow = (w as f64 * ratio).floor();
                let oh = (h as f64 * ratio).floor();
                if !ow.is_finite()
                    || !oh.is_finite()
                    || ow < 1.0
                    || oh < 1.0
                    || ow > u32::MAX as f64
                    || oh > u32::MAX as f64
                {
                    return Err(invalid_argument(
                        "random shortest size output dimensions must be positive and fit u32",
                    ));
                }
                return ResizeConfig::with_interpolation(ow as u32, oh as u32, *interpolation)
                    .apply(ImageSample::Decoded(s));
            }
            Self::ScaleJitter {
                target_width,
                target_height,
                scale_range: (a, b),
                interpolation,
            } => {
                let ratio = (*target_height as f64 / h as f64).min(*target_width as f64 / w as f64)
                    * (a + rng.next_f64() * (b - a));
                let ow = (w as f64 * ratio).floor();
                let oh = (h as f64 * ratio).floor();
                if !ow.is_finite()
                    || !oh.is_finite()
                    || ow < 1.0
                    || oh < 1.0
                    || ow > u32::MAX as f64
                    || oh > u32::MAX as f64
                {
                    return Err(invalid_argument(
                        "scale jitter output dimensions must be positive and fit u32",
                    ));
                }
                return ResizeConfig::with_interpolation(ow as u32, oh as u32, *interpolation)
                    .apply(ImageSample::Decoded(s));
            }
            Self::GaussianNoise { mean, sigma, clip } => {
                s.image = super::gaussian_noise(&s.image, *mean, *sigma, *clip, rng)?;
            }
            Self::BiasField(config) => {
                let input = match axis {
                    ImageAxisOrder::Hwc => s.image,
                    ImageAxisOrder::Chw => s.image.permute(&[1, 2, 0])?,
                };
                let image = config.apply(&input, [1, h, w], c, rng)?;
                s.image = match axis {
                    ImageAxisOrder::Hwc => image,
                    ImageAxisOrder::Chw => image.permute(&[2, 0, 1])?,
                };
            }
            Self::GibbsRinging(config) => {
                let input = match axis {
                    ImageAxisOrder::Hwc => s.image,
                    ImageAxisOrder::Chw => s.image.permute(&[1, 2, 0])?,
                };
                let image = config.apply(&input, [1, h, w], c, rng)?;
                s.image = match axis {
                    ImageAxisOrder::Hwc => image,
                    ImageAxisOrder::Chw => image.permute(&[2, 0, 1])?,
                };
            }
            Self::RicianNoise {
                std_range,
                per_channel,
            } => {
                let input = match axis {
                    ImageAxisOrder::Hwc => s.image,
                    ImageAxisOrder::Chw => s.image.permute(&[1, 2, 0])?,
                };
                let image = super::noise::rician_noise(&input, c, *std_range, *per_channel, rng)?;
                s.image = match axis {
                    ImageAxisOrder::Hwc => image,
                    ImageAxisOrder::Chw => image.permute(&[2, 0, 1])?,
                };
            }
            Self::ShotNoise { intensity } => {
                s.image = super::shot_noise(&s.image, *intensity, rng)?;
            }
            Self::RandomToneCurve { .. }
            | Self::Clahe { .. }
            | Self::AdditiveNoise(_)
            | Self::MultiplicativeNoise { .. }
            | Self::RandomSunFlare(_)
            | Self::HueSaturationValue(_)
            | Self::Halftone { .. }
            | Self::Dithering(_)
            | Self::RingingOvershoot { .. }
            | Self::Enhance { .. }
            | Self::Emboss { .. }
            | Self::ExposureMatching { .. }
            | Self::FancyPca { .. }
            | Self::Colorize(_)
            | Self::ToSepia { .. }
            | Self::RgbShift(_)
            | Self::RandomBrightnessContrast { .. }
            | Self::RandomGamma { .. }
            | Self::RandomFog(_)
            | Self::AtmosphericFog(_)
            | Self::Spatter(_)
            | Self::RandomGravel(_)
            | Self::RandomSnow(_)
            | Self::RandomRain(_)
            | Self::LensFlare(_)
            | Self::RandomShadow(_)
            | Self::PlasmaShadow { .. }
            | Self::PlasmaBrightnessContrast { .. }
            | Self::Illumination { .. }
            | Self::FilmGrain { .. }
            | Self::IsoNoise { .. }
            | Self::GlassBlur { .. }
            | Self::ZoomBlur { .. }
            | Self::StochasticConvolution { .. }
            | Self::AdvancedBlur(_)
            | Self::Blur { .. }
            | Self::MedianBlur
            | Self::PlanckianJitter { .. }
            | Self::Vignetting { .. }
            | Self::ChromaticAberration { .. }
                if axis != ImageAxisOrder::Hwc || h == 0 || w == 0 =>
            {
                return Err(invalid_argument(format!(
                    "{} requires nonempty HWC input",
                    self.name()
                )));
            }
            Self::Blur {
                kernel_range: (low, high),
            } => {
                let size =
                    low + 2 * rng.gen_range_usize(0..((high - low) / 2 + 1) as usize)? as u32;
                s.image = super::blur::box_blur(&s.image, size as usize)?;
            }
            Self::AdvancedBlur(config) => {
                let AdvancedBlurConfig {
                    kernel_range,
                    sigma_x_range,
                    sigma_y_range,
                    rotate_range,
                    beta_range,
                    noise_range,
                } = config.as_ref();
                let (low, high) = kernel_range;
                let size =
                    low + 2 * rng.gen_range_usize(0..((high - low) / 2 + 1) as usize)? as u32;
                let mut sample = |(low, high)| low + rng.next_f64() * (high - low);
                let sigma = (sample(*sigma_x_range), sample(*sigma_y_range));
                let angle = sample(*rotate_range);
                let beta = sample(*beta_range);
                s.image = super::blur::advanced_blur(
                    &s.image,
                    size as usize,
                    sigma,
                    angle,
                    beta,
                    *noise_range,
                    rng,
                )?;
            }
            Self::StochasticConvolution {
                kernel_range: (low, high),
                strength_range,
                per_channel,
                padding_mode,
            } => {
                let size =
                    low + 2 * rng.gen_range_usize(0..((high - low) / 2 + 1) as usize)? as u32;
                let (low, high) = (f64::from(strength_range.0), f64::from(strength_range.1));
                let strength = low + rng.next_f64() * (high - low);
                s.image = super::blur::stochastic_convolution(
                    &s.image,
                    size as usize,
                    strength,
                    *per_channel,
                    *padding_mode,
                    rng,
                )?;
            }
            Self::ZoomBlur {
                max_factor_range,
                step_factor_range,
            } => {
                let mut sample = |(low, high)| low + rng.next_f64() * (high - low);
                let max_factor = sample(*max_factor_range);
                let step = sample(*step_factor_range);
                s.image = super::blur::zoom_blur(&s.image, max_factor, step)?;
            }
            Self::GlassBlur {
                sigma,
                max_delta,
                iterations,
            } => {
                s.image = super::blur::glass_blur(
                    &s.image,
                    *sigma,
                    *max_delta as usize,
                    *iterations,
                    rng,
                )?;
            }
            Self::IsoNoise {
                color_shift_range,
                intensity_range,
            } => {
                let mut sample = |(low, high)| low + rng.next_f64() * (high - low);
                let color_shift = sample(*color_shift_range);
                let intensity = sample(*intensity_range);
                s.image = super::noise::iso_noise(&s.image, color_shift, intensity, rng)?;
            }
            Self::FilmGrain {
                intensity_range: (low, high),
                grain_size_range: (min, max),
                per_channel,
            } => {
                let intensity = low + rng.next_f64() * (high - low);
                let size = *min as usize + rng.gen_range_usize(0..((*max - *min) as usize + 1))?;
                s.image = super::noise::film_grain(&s.image, intensity, size, *per_channel, rng)?;
            }
            Self::MultiplicativeNoise {
                multiplier,
                per_channel,
                elementwise,
            } => {
                s.image = super::noise::multiplicative_noise(
                    &s.image,
                    *multiplier,
                    *per_channel,
                    *elementwise,
                    rng,
                )?;
            }
            Self::AdditiveNoise(config) => {
                s.image = super::noise::additive_noise(
                    &s.image,
                    &config.distribution,
                    config.sharing,
                    rng,
                )?;
            }
            Self::Illumination {
                intensity_range,
                mode,
                effect,
            } => {
                s.image = super::local_color::illumination(
                    &s.image,
                    *intensity_range,
                    *mode,
                    *effect,
                    rng,
                )?;
            }
            Self::PlasmaShadow {
                shadow_intensity_range,
                plasma_size,
                roughness,
            } => {
                let mut sample = |(a, b)| a + rng.next_f64() * (b - a);
                let amount = sample(*shadow_intensity_range);
                s.image = super::local_color::apply_plasma(
                    &s.image,
                    self.name(),
                    *plasma_size as usize,
                    *roughness,
                    amount == 0.0,
                    rng,
                    |t| (1.0 - t * amount, 0.0),
                )?;
            }
            Self::PlasmaBrightnessContrast {
                brightness_range,
                contrast_range,
                plasma_size,
                roughness,
            } => {
                let mut sample = |(a, b)| a + rng.next_f64() * (b - a);
                let (brightness, contrast) = (sample(*brightness_range), sample(*contrast_range));
                s.image = super::local_color::apply_plasma(
                    &s.image,
                    self.name(),
                    *plasma_size as usize,
                    *roughness,
                    brightness == 0.0 && contrast == 0.0,
                    rng,
                    |t| (1.0 + t * contrast, t * (brightness - 0.5 * contrast)),
                )?;
            }
            Self::RandomShadow(config) => {
                s.image = super::local_color::random_shadow(&s.image, config, rng)?;
            }
            Self::RandomSunFlare(config) => {
                s.image = super::flare::random_sun_flare(&s.image, config, rng)?;
            }
            Self::LensFlare(config) => {
                s.image = super::flare::lens_flare(&s.image, config, rng)?;
            }
            Self::RandomRain(config) => {
                s.image = super::weather::random_rain(&s.image, config, rng)?;
            }
            Self::RandomBrightnessContrast {
                brightness_range,
                contrast_range,
                brightness_by_max,
                ensure_safe_output,
            } => {
                let mut sample = |(lo, hi)| lo + rng.next_f64() * (hi - lo);
                let brightness = sample(*brightness_range);
                let contrast = sample(*contrast_range);
                s.image = super::color::random_brightness_contrast(
                    &s.image,
                    brightness,
                    contrast,
                    *brightness_by_max,
                    *ensure_safe_output,
                )?;
            }
            Self::HueSaturationValue(ranges) => {
                let shifts = ranges.map(|(lo, hi)| lo + rng.next_f64() * (hi - lo));
                s.image = super::color::hue_saturation_value(&s.image, shifts)?;
            }
            Self::ExposureMatching {
                target_mean_range: (lo, hi),
                gain_range,
            } => {
                let target = lo + rng.next_f64() * (hi - lo);
                s.image = super::color::exposure_matching(&s.image, target, *gain_range)?;
            }
            Self::Emboss {
                alpha_range,
                strength_range,
            } => {
                let mut sample = |(lo, hi)| lo + rng.next_f64() * (hi - lo);
                s.image =
                    super::blur::emboss(&s.image, sample(*alpha_range), sample(*strength_range))?;
            }
            Self::Enhance {
                mode,
                alpha_range: (lo, hi),
            } => {
                let alpha = lo + rng.next_f64() * (hi - lo);
                s.image = super::blur::enhance(&s.image, *mode, alpha)?;
            }
            Self::RingingOvershoot {
                blur_range: (lo, hi),
                cutoff_range: (min, max),
            } => {
                let size =
                    *lo as usize + 2 * rng.gen_range_usize(0..((hi - lo) / 2 + 1) as usize)?;
                let cutoff = min + rng.next_f64() * (max - min);
                s.image = super::blur::ringing_overshoot(&s.image, size, cutoff)?;
            }
            Self::Halftone {
                dot_size_range: (lo, hi),
                blend_range: (min, max),
            } => {
                let size = *lo as usize + rng.gen_range_usize(0..(*hi - *lo) as usize + 1)?;
                let blend = min + rng.next_f64() * (max - min);
                s.image = super::dither::halftone(&s.image, size, blend)?;
            }
            Self::Dithering(config) => {
                s.image = config.apply(&s.image, rng)?;
            }
            Self::FancyPca { alpha } => {
                s.image = super::pca::fancy_pca(&s.image, *alpha, rng)?;
            }
            Self::Colorize(config) => {
                s.image = config.apply(&s.image, rng)?;
            }
            Self::ToSepia { alpha } => {
                s.image = super::color::to_sepia(&s.image, *alpha)?;
            }
            Self::RgbShift(ranges) => {
                s.image = super::noise::rgb_shift(&s.image, ranges, rng)?;
            }
            Self::RandomGamma {
                gamma_range: (lo, hi),
            } => {
                let exponent = (lo + rng.next_f64() * (hi - lo)) / 100.0;
                s.image = super::color::random_gamma(&s.image, exponent)?;
            }
            Self::RandomFog(config) => {
                s.image = super::fog::random_fog(&s.image, config, rng)?;
            }
            Self::AtmosphericFog(config) => {
                s.image = super::fog::atmospheric_fog(&s.image, config, rng)?;
            }
            Self::Spatter(config) => {
                s.image = super::weather::spatter(&s.image, config, rng)?;
            }
            Self::RandomGravel(config) => {
                s.image = super::weather::random_gravel(&s.image, config, rng)?;
            }
            Self::RandomSnow(config) => {
                s.image = super::weather::random_snow(&s.image, config, rng)?;
            }
            Self::MedianBlur => {
                s.image = super::blur::median_blur_3x3(&s.image)?;
            }
            Self::RandomToneCurve { scale } => {
                s = require_u8_hwc(s, self.name())?;
                s.image = super::local_color::tone_curve(&s.image, *scale, rng)?;
            }
            Self::Vignetting {
                intensity_range,
                center_range,
            } => {
                let mut sample = |(low, high)| low + rng.next_f64() * (high - low);
                let intensity = sample(*intensity_range);
                let center = (sample(*center_range), sample(*center_range));
                s.image = super::local_color::vignette(&s.image, intensity, center)?;
            }
            Self::PlanckianJitter {
                temperature_range: (low, high),
            } => {
                let kelvin = low + rng.next_f64() * (high - low);
                s.image = super::local_color::planckian_jitter(&s.image, kelvin)?;
            }
            Self::Clahe { grid, clip_limit } => {
                s.image = super::local_color::clahe(&s.image, *grid, *clip_limit)?;
            }
            Self::ChromaticAberration {
                red_range,
                blue_range,
                interpolation,
            } => {
                let mut coefficient = |(low, high)| low + rng.next_f64() * (high - low);
                let red = coefficient(*red_range);
                let blue = coefficient(*blue_range);
                s.image = super::advanced_geometry::chromatic_aberration(
                    &s.image,
                    red,
                    blue,
                    *interpolation,
                )?;
            }
            Self::PadIfNeeded(config) => {
                let pad = config.sample(h, w, rng)?;
                return pad.apply(ImageSample::Decoded(s), axis);
            }
            Self::LetterBox(config) => {
                if axis != ImageAxisOrder::Hwc {
                    return Err(invalid_argument("LetterBox requires uint8 HWC RGB input"));
                }
                let (resize, pad) = config.resolve(w, h)?;
                let resized = resize.apply(ImageSample::Decoded(s))?;
                return pad.apply(resized, axis);
            }
            Self::Defocus {
                radius_range: (low, high),
            } => {
                if axis != ImageAxisOrder::Hwc || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "Defocus requires nonempty uint8 HWC RGB input",
                    ));
                }
                let radius = rng.gen_range_usize(*low as usize..*high as usize + 1)? as u32;
                s.image = super::blur::defocus(&s.image, radius)?;
            }
            Self::CoarseDropout {
                holes,
                hole_height,
                hole_width,
                fill,
            } => {
                s.image = super::pixel_mask::coarse_dropout(
                    &s.image,
                    axis,
                    (h, w),
                    *holes,
                    *hole_height,
                    *hole_width,
                    *fill,
                    rng,
                )?;
            }
            Self::MotionBlur {
                length_range: (low, high),
                angle_range: (start, end),
            } => {
                if axis != ImageAxisOrder::Hwc || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "MotionBlur requires nonempty uint8 HWC RGB input",
                    ));
                }
                let length =
                    low + 2 * rng.gen_range_usize(0..((high - low) / 2 + 1) as usize)? as u32;
                let angle = (start + rng.next_f64() * (end - start)).rem_euclid(180.0);
                s.image = super::blur::motion_blur(&s.image, length, angle)?;
            }
            Self::UnsharpMask {
                sigma,
                amount,
                threshold,
            } => {
                if axis != ImageAxisOrder::Hwc || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "UnsharpMask requires nonempty uint8 HWC RGB input",
                    ));
                }
                s = require_u8_hwc(s, self.name())?;
                s.image = super::blur::unsharp_mask(s.clone(), *sigma, *amount, *threshold)?;
            }
            Self::Downscale {
                scale_range: (low, high),
                down_interpolation,
                up_interpolation,
            } => {
                let s = require_u8_hwc(s, self.name())?;
                if axis != ImageAxisOrder::Hwc || c != 3 || w == 0 || h == 0 {
                    return Err(invalid_argument(
                        "Downscale requires a nonempty uint8 HWC RGB image",
                    ));
                }
                let width =
                    u32::try_from(w).map_err(|_| invalid_shape("Downscale width exceeds u32"))?;
                let height =
                    u32::try_from(h).map_err(|_| invalid_shape("Downscale height exceeds u32"))?;
                let scale = low + rng.next_f64() * (high - low);
                let small_width = (f64::from(width) * scale) as u32;
                let small_height = (f64::from(height) * scale) as u32;
                let reduced = ResizeConfig::with_interpolation(
                    small_width.max(1),
                    small_height.max(1),
                    *down_interpolation,
                )
                .apply(ImageSample::Decoded(s))?;
                return ResizeConfig::with_interpolation(width, height, *up_interpolation)
                    .apply(reduced);
            }
            Self::SaltAndPepper { amount, salt_ratio } => {
                s.image = super::pixel_mask::salt_and_pepper(
                    &s.image,
                    axis,
                    (h, w),
                    *amount,
                    *salt_ratio,
                    rng,
                )?;
            }
            Self::ChannelDropout(config) => {
                s.image = config.apply(&s.image, axis, rng)?;
            }
            Self::SquareSymmetry(element) => {
                let (reflect, turns) = SquareSymmetry::sample(*element, rng)?.components();
                if reflect {
                    let order = match axis {
                        ImageAxisOrder::Hwc => [1, 0, 2],
                        ImageAxisOrder::Chw => [0, 2, 1],
                    };
                    s.image = s.image.permute(&order)?;
                }
                s.image = super::quarter_turn(&s.image, axis, turns)?;
            }
            Self::Fda(config) => {
                if axis != ImageAxisOrder::Hwc {
                    return Err(invalid_argument("FDA requires HWC input"));
                }
                s.image = config.apply(&s.image, rng)?;
            }
            Self::PixelDistributionAdaptation(config) => {
                if axis != ImageAxisOrder::Hwc {
                    return Err(invalid_argument(
                        "PixelDistributionAdaptation requires HWC input",
                    ));
                }
                s.image = config.apply(&s.image, rng)?;
            }
            Self::HistogramMatching(config) => {
                if axis != ImageAxisOrder::Hwc {
                    return Err(invalid_argument("HistogramMatching requires HWC input"));
                }
                s.image = config.apply(&s.image, rng)?;
            }
            Self::WaterRefraction(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "WaterRefraction requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng)?;
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        config.interpolation,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::RandomGridShuffle(config) => {
                s.image = config.sample(h, w, rng)?.apply(&s.image, axis)?;
            }
            Self::PixelSpread(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "PixelSpread requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng)?;
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        InterpolationMode::Nearest,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::ThinPlateSpline(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "ThinPlateSpline requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        config.interpolation,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::GridElasticDeform(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "GridElasticDeform requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        config.interpolation,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::PiecewiseAffine(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "PiecewiseAffine requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng)?;
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        config.interpolation,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::GridDistortion(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "GridDistortion requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        config.interpolation,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::OpticalDistortion(config) => {
                use super::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "OpticalDistortion requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return super::remap::remap_image(
                        ImageSample::Decoded(s),
                        axis,
                        (h, w),
                        config.interpolation,
                        config.fill,
                        |x, y| map.source([x as f64, y as f64]),
                    );
                }
            }
            Self::Superpixels(config) => {
                if axis != ImageAxisOrder::Hwc {
                    return Err(invalid_argument("Superpixels requires HWC input"));
                }
                s.image = config.apply(&s.image, rng)?;
            }
            Self::Morphological(config) => {
                s.image = config.apply(&s.image, axis)?;
            }
            Self::SafeRotate(config) => {
                if h == 0 || w == 0 {
                    return Err(invalid_argument("SafeRotate requires a nonempty image"));
                }
                let matrix = config.sample(w, h, rng);
                return config.apply(ImageSample::Decoded(s), axis, matrix);
            }
            Self::RandomRotate90 => {
                let turns = rng.gen_range_usize(0..4)?;
                s.image = super::quarter_turn(&s.image, axis, turns)?;
            }
            Self::Transpose => {
                let order = match axis {
                    ImageAxisOrder::Hwc => [1, 0, 2],
                    ImageAxisOrder::Chw => [0, 2, 1],
                };
                s.image = s.image.permute(&order)?;
            }
            Self::GridMask(config) => {
                s.image = config.apply(&s.image, axis, config.sample(h, w, rng)?)?;
            }
            Self::XyMasking(config) => {
                s.image = config.apply(&s.image, axis, &config.sample(h, w, rng)?)?;
            }
            Self::PixelDropout(config) => {
                s.image = config.apply(&s.image, axis, rng, None)?;
            }
            Self::GridDropout {
                unit_size,
                ratio,
                fill,
            } => {
                s.image = super::pixel_mask::grid_dropout(
                    &s.image,
                    axis,
                    (h, w),
                    *unit_size,
                    *ratio,
                    *fill,
                    rng,
                )?;
            }
            Self::RandomScale {
                range,
                y,
                interpolation,
            } => {
                s = require_u8_hwc(s, self.name())?;
                let (w, h) = super::sample_scale(w, h, *range, *y, rng)?;
                return ResizeConfig::with_interpolation(w, h, *interpolation)
                    .apply(ImageSample::Decoded(s));
            }
            Self::LongestMaxSize(config) => {
                s = require_u8_hwc(s, self.name())?;
                let (w, h) = config.sample(w, h, rng)?;
                return ResizeConfig::with_interpolation(w, h, config.interpolation)
                    .apply(ImageSample::Decoded(s));
            }
            Self::CropAndPad(config) => {
                let ([x, y, cw, ch], pad) = config.sample(h, w, rng)?;
                let mut result = ImageSample::Decoded(s);
                if (x, y, cw as usize, ch as usize) != (0, 0, w, h) {
                    result = CropConfig::new(x, y, cw, ch).apply(result, axis)?;
                }
                if pad.left != 0 || pad.right != 0 || pad.top != 0 || pad.bottom != 0 {
                    result = pad.apply(result, axis)?;
                }
                if config.keep_size
                    && (cw as usize + pad.left as usize + pad.right as usize != w
                        || ch as usize + pad.top as usize + pad.bottom as usize != h)
                {
                    if axis == ImageAxisOrder::Chw {
                        result = LayoutConfig::new(ImageAxisOrder::Hwc).apply(result, axis)?;
                    }
                    result =
                        ResizeConfig::with_interpolation(w as u32, h as u32, config.interpolation)
                            .apply(result)?;
                    if axis == ImageAxisOrder::Chw {
                        result = LayoutConfig::new(axis).apply(result, ImageAxisOrder::Hwc)?;
                    }
                }
                return Ok(result);
            }
            Self::Jpeg {
                quality: (low, high),
            } => {
                let q = rng.gen_range_usize(*low as usize..*high as usize + 1)? as u8;
                s.image = super::v2_jpeg::jpeg(&s.image, axis, q)?;
            }
            Self::PermuteChannels(order) => {
                if order.len() != c
                    || order.iter().any(|i| *i >= c)
                    || (0..c).any(|i| !order.contains(&i))
                {
                    return Err(invalid_argument("channel order must be a permutation"));
                }
                s.image = permute(&s.image, &dims, axis, order)?;
            }
            Self::RandomChannelPermutation { probability } => {
                if rng.gen_bool(*probability)? {
                    let mut order: Vec<_> = (0..c).collect();
                    for i in (1..c).rev() {
                        let j = rng.gen_range_usize(0..i + 1)?;
                        order.swap(i, j);
                    }
                    s.image = permute(&s.image, &dims, axis, &order)?;
                }
            }
            Self::Rgb => {
                s.image = convert_rgb(&s.image, axis, h, w, c)?;
            }
            Self::ToDtype { .. } => unreachable!("ToDtype handled before image dispatch"),
        }
        Ok(ImageSample::Decoded(s))
    }
}
fn permute(
    input: &Tensor,
    dims: &[usize],
    axis: ImageAxisOrder,
    order: &[usize],
) -> RivetResult<Tensor> {
    let (h, w) = match axis {
        ImageAxisOrder::Hwc => (dims[0], dims[1]),
        ImageAxisOrder::Chw => (dims[1], dims[2]),
    };
    input
        .with_cpu_storage(|storage, layout| {
            macro_rules! run {
                ($values:expr,$t:ty) => {{
                    let src = $values;
                    let st = layout.stride();
                    let base = layout.start_offset();
                    Tensor::from_exact_writer::<$t, _, _>(dims, input.device(), |out| {
                        match axis {
                            ImageAxisOrder::Hwc => {
                                for y in 0..h {
                                    for x in 0..w {
                                        let offset = base + y * st[0] + x * st[1];
                                        if order.len() == 3 {
                                            out.write_next(src[offset + order[0] * st[2]])?;
                                            out.write_next(src[offset + order[1] * st[2]])?;
                                            out.write_next(src[offset + order[2] * st[2]])?;
                                        } else {
                                            for &c in order {
                                                out.write_next(src[offset + c * st[2]])?;
                                            }
                                        }
                                    }
                                }
                            }
                            ImageAxisOrder::Chw => {
                                for &c in order {
                                    for y in 0..h {
                                        for x in 0..w {
                                            out.write_next(
                                                src[base + c * st[0] + y * st[1] + x * st[2]],
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(v) => run!(v, u8),
                CpuStorageRef::F32(v) => run!(v, f32),
                _ => Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                }),
            }
        })
        .map_err(Into::into)
}

fn convert_rgb(
    input: &Tensor,
    axis: ImageAxisOrder,
    height: usize,
    width: usize,
    channels: usize,
) -> RivetResult<Tensor> {
    if !(1..=4).contains(&channels) {
        return Err(invalid_argument("RGB requires 1, 2, 3, or 4 channels"));
    }
    if !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument("RGB requires uint8 or float32 pixels"));
    }
    if channels == 3 {
        return Ok(input.clone());
    }
    let dims = match axis {
        ImageAxisOrder::Hwc => [height, width, 3],
        ImageAxisOrder::Chw => [3, height, width],
    };
    input
        .with_cpu_storage(|storage, layout| {
            macro_rules! convert {
                ($values:expr, $dtype:ty) => {{
                    let values = $values;
                    let stride = layout.stride();
                    let start = layout.start_offset();
                    Tensor::from_exact_writer::<$dtype, _, _>(dims, input.device(), |output| {
                        match axis {
                            ImageAxisOrder::Hwc => {
                                for y in 0..height {
                                    for x in 0..width {
                                        let offset = start + y * stride[0] + x * stride[1];
                                        if channels <= 2 {
                                            let value = *values
                                                .get(offset)
                                                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                                            output.write_next(value)?;
                                            output.write_next(value)?;
                                            output.write_next(value)?;
                                        } else {
                                            for channel in 0..3 {
                                                output.write_next(
                                                    *values
                                                        .get(offset + channel * stride[2])
                                                        .ok_or(
                                                            rivet_core::Error::StorageOutOfBounds,
                                                        )?,
                                                )?;
                                            }
                                        }
                                    }
                                }
                            }
                            ImageAxisOrder::Chw => {
                                for channel in 0..3 {
                                    let source_channel = if channels <= 2 { 0 } else { channel };
                                    let channel_offset = start + source_channel * stride[0];
                                    for y in 0..height {
                                        let row_offset = channel_offset + y * stride[1];
                                        for x in 0..width {
                                            output.write_next(
                                                *values
                                                    .get(row_offset + x * stride[2])
                                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(values) => convert!(values, u8),
                CpuStorageRef::F32(values) => convert!(values, f32),
                _ => unreachable!("validated RGB dtype"),
            }
        })
        .map_err(Into::into)
}

#[cfg(test)]
mod rgb_tests {
    use super::*;

    #[test]
    fn rgb_modes_preserve_layout_and_strided_offsets() {
        for channels in 1..=4 {
            let data: Vec<u8> = (0..2 * 3 * channels).map(|value| value as u8).collect();
            let image = Tensor::from_vec(data.clone(), [2, 3, channels], &Device::Cpu).unwrap();
            let cropped = image.narrow(1, 1, 2).unwrap();
            for axis in [ImageAxisOrder::Hwc, ImageAxisOrder::Chw] {
                let input = match axis {
                    ImageAxisOrder::Hwc => cropped.clone(),
                    ImageAxisOrder::Chw => cropped.permute(&[2, 0, 1]).unwrap(),
                };
                let output = convert_rgb(&input, axis, 2, 2, channels).unwrap();
                let hwc = match axis {
                    ImageAxisOrder::Hwc => output,
                    ImageAxisOrder::Chw => output.permute(&[1, 2, 0]).unwrap(),
                };
                let expected: Vec<u8> = (0..2)
                    .flat_map(|y| {
                        let data = &data;
                        (1..3).flat_map(move |x| {
                            (0..3).map(move |channel| {
                                data[(y * 3 + x) * channels
                                    + if channels <= 2 { 0 } else { channel }]
                            })
                        })
                    })
                    .collect();
                assert_eq!(hwc.to_vec::<u8>().unwrap(), expected);
                let f32_base = image.to_dtype(DType::F32).unwrap().narrow(1, 1, 2).unwrap();
                let f32_input = match axis {
                    ImageAxisOrder::Hwc => f32_base,
                    ImageAxisOrder::Chw => f32_base.permute(&[2, 0, 1]).unwrap(),
                };
                let output = convert_rgb(&f32_input, axis, 2, 2, channels).unwrap();
                let hwc = match axis {
                    ImageAxisOrder::Hwc => output,
                    ImageAxisOrder::Chw => output.permute(&[1, 2, 0]).unwrap(),
                };
                assert_eq!(
                    hwc.to_vec::<f32>().unwrap(),
                    expected.into_iter().map(f32::from).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn rgb_identity_shares_storage_and_rejects_unknown_modes() {
        let image = Tensor::from_vec(vec![1u8, 2, 3], [1, 1, 3], &Device::Cpu).unwrap();
        let output = convert_rgb(&image, ImageAxisOrder::Hwc, 1, 1, 3).unwrap();
        assert!(output.same_storage(&image));
        let invalid = Tensor::from_vec(vec![0u8; 5], [1, 1, 5], &Device::Cpu).unwrap();
        assert!(convert_rgb(&invalid, ImageAxisOrder::Hwc, 1, 1, 5).is_err());
    }
}

#[cfg(test)]
mod v2_tests {
    use super::*;
    fn run(config: V2Config, image: Tensor, axis: ImageAxisOrder, seed: u64) -> Tensor {
        let mut rng = RandomStream::from_seed(seed);
        config
            .apply(
                ImageSample::Decoded(DecodedSample { image, label: 17 }),
                axis,
                &mut rng,
            )
            .unwrap()
            .into_decoded()
            .unwrap()
            .image
    }
    #[test]
    fn channel_permutations_read_strided_u8_and_float_without_changing_layout() {
        let image = Tensor::from_vec((0..36u8).collect::<Vec<_>>(), [3, 4, 3], &Device::Cpu)
            .unwrap()
            .narrow(1, 1, 2)
            .unwrap();
        for axis in [ImageAxisOrder::Hwc, ImageAxisOrder::Chw] {
            let input = if axis == ImageAxisOrder::Chw {
                image.permute(&[2, 0, 1]).unwrap()
            } else {
                image.clone()
            };
            let output = run(
                V2Config::PermuteChannels(vec![2, 0, 1]),
                input.clone(),
                axis,
                4,
            );
            let expected = if axis == ImageAxisOrder::Chw {
                vec![
                    5, 8, 17, 20, 29, 32, 3, 6, 15, 18, 27, 30, 4, 7, 16, 19, 28, 31,
                ]
            } else {
                vec![
                    5, 3, 4, 8, 6, 7, 17, 15, 16, 20, 18, 19, 29, 27, 28, 32, 30, 31,
                ]
            };
            assert_eq!(
                output.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
                expected
            );
            let float = run(
                V2Config::PermuteChannels(vec![2, 0, 1]),
                input.to_dtype(DType::F32).unwrap(),
                axis,
                4,
            );
            assert_eq!(
                float.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.into_iter().map(f32::from).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn photometric_fixed_ranges_match_existing_color_sequence_and_sampling_order() {
        let image =
            Tensor::from_vec(vec![15u8, 40, 80, 100, 150, 210], [1, 2, 3], &Device::Cpu).unwrap();
        let config = V2Config::RandomPhotometricDistort {
            brightness: (0.9, 0.9),
            contrast: (1.1, 1.1),
            saturation: (0.8, 0.8),
            hue: (0.02, 0.02),
            probability: 1.0,
        };
        let seed = 8;
        let actual = run(config.clone(), image.clone(), ImageAxisOrder::Hwc, seed);
        let mut rng = RandomStream::from_seed(seed);
        for _ in 0..4 {
            assert!(rng.gen_bool(1.0).unwrap());
            rng.next_f64();
        }
        let before = rng.gen_bool(0.5).unwrap();
        assert!(rng.gen_bool(1.0).unwrap());
        let mut order = vec![0, 1, 2];
        for i in (1..3).rev() {
            let j = rng.gen_range_usize(0..i + 1).unwrap();
            order.swap(i, j);
        }
        let mut expected = ImageSample::Decoded(DecodedSample { image, label: 17 });
        expected = BrightnessConfig::new(0.9).apply(expected).unwrap();
        if before {
            expected = ContrastConfig::new(1.1).apply(expected).unwrap();
        }
        expected = SaturationConfig::new(0.8).apply(expected).unwrap();
        expected = HueConfig::new(0.02).apply(expected).unwrap();
        if !before {
            expected = ContrastConfig::new(1.1).apply(expected).unwrap();
        }
        let image = expected.into_decoded().unwrap().image;
        let expected = permute(&image, image.dims(), ImageAxisOrder::Hwc, &order).unwrap();
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<u8>().unwrap()
        );
    }
    #[test]
    fn zoom_out_uses_floor_offset_and_per_channel_fill() {
        let image = Tensor::from_vec(vec![5u8, 6, 7, 8, 9, 10], [1, 2, 3], &Device::Cpu).unwrap();
        let seed = 7;
        let mut rng = RandomStream::from_seed(seed);
        assert!(rng.gen_bool(1.0).unwrap());
        rng.next_f64();
        let left = (2.0 * rng.next_f64()).floor() as u32;
        let top = (1.0 * rng.next_f64()).floor() as u32;
        let expected = PadConfig::with_sides(left, top, 2 - left, 1 - top)
            .with_fill_channels([20.0, 30.0, 40.0])
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: image.clone(),
                    label: 17,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap()
            .image;
        let actual = run(
            V2Config::RandomZoomOut {
                fill: vec![20.0, 30.0, 40.0],
                side_range: (2.0, 2.0),
                probability: 1.0,
            },
            image,
            ImageAxisOrder::Hwc,
            seed,
        );
        assert_eq!(actual.dims(), [2, 4, 3]);
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<u8>().unwrap()
        );
    }
    #[test]
    fn probability_zero_preserves_storage_and_invalid_parameters_are_rejected() {
        let image = Tensor::from_vec(vec![1u8, 2, 3], [1, 1, 3], &Device::Cpu).unwrap();
        for config in [
            V2Config::RandomChannelPermutation { probability: 0.0 },
            V2Config::RandomZoomOut {
                fill: vec![0.0],
                side_range: (1.0, 4.0),
                probability: 0.0,
            },
            V2Config::RandomPhotometricDistort {
                brightness: (0.8, 1.2),
                contrast: (0.5, 1.5),
                saturation: (0.5, 1.5),
                hue: (-0.1, 0.1),
                probability: 0.0,
            },
        ] {
            let actual = run(config, image.clone(), ImageAxisOrder::Hwc, 1);
            assert!(actual.same_storage(&image));
        }
        for config in [
            V2Config::RandomResize {
                min_size: 2,
                max_size: 2,
                interpolation: InterpolationMode::Bilinear,
            },
            V2Config::RandomPhotometricDistort {
                brightness: (f32::NAN, 1.0),
                contrast: (0.5, 1.5),
                saturation: (0.5, 1.5),
                hue: (-0.1, 0.1),
                probability: 0.5,
            },
            V2Config::RandomZoomOut {
                fill: vec![1.0, 2.0],
                side_range: (1.0, 2.0),
                probability: 0.5,
            },
        ] {
            assert!(config.validate().is_err());
        }
    }
}

#[cfg(test)]
mod batch_contract_tests {
    use super::*;
    #[test]
    fn unscaled_batch_cast_requires_rank_four() {
        let input = Tensor::from_vec(vec![1u8; 3], [1, 1, 3], &Device::Cpu).unwrap();
        let config = V2Config::ToDtype {
            dtype: DType::F32,
            scale: false,
        };
        assert!(config.apply_batch(input, ImageAxisOrder::Hwc).is_err());
    }
}
