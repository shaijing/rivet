use super::V2Config;
use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::{AdvancedBlurConfig, InterpolationMode, degradation};
use rivet_core::DType;

impl V2Config {
    pub fn validate(&self) -> RivetResult<()> {
        let valid = match self {
            Self::AnnotationArtifacts(config) => config.valid(),
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
            Self::LensFlare(config) => degradation::flare::lens_flare_valid(config),
            Self::RandomSunFlare(config) => degradation::flare::random_sun_flare_valid(config),
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
                return degradation::noise::validate_gaussian_noise(*mean, *sigma);
            }
            Self::BiasField(config) => return config.validate(),
            Self::GibbsRinging(config) => return config.validate(),
            Self::HEStain(config) => return config.validate(),
            Self::KSpaceSpikeNoise(config) => return config.validate(),
            Self::RicianNoise { std_range, .. } => {
                return degradation::noise::validate_rician_std(*std_range);
            }
            Self::ShotNoise { intensity } => {
                return degradation::noise::validate_shot_noise(*intensity);
            }
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
            Self::Webp { quality: (a, b) } | Self::Jpeg { quality: (a, b) } => {
                *a >= 1 && b >= a && *b <= 100
            }
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
}
