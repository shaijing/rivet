use super::op::{BatchConfig, ImageOp, IndexOp, SourceOp};
use super::transform::TransformSequence;
use crate::runtime::RuntimeConfig;
use crate::sample::image::EncodedImageSample;
use crate::source::ImageSource;
use crate::transforms::{
    ArbitraryRotateConfig, ElasticTransformConfig, InterpolationMode, PerspectiveConfig, Point2,
    RandomAffineConfig, RandomErasingConfig, RandomPerspectiveConfig, RandomResizedCropConfig,
};
use rivet_core::DType;
use rivet_data::dataset::Dataset;
use rivet_plan::DeviceTarget;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceCutPlacement {
    pub(crate) after_ops: usize,
    pub(crate) after_batch: bool,
    pub(crate) target: DeviceTarget,
}

/// Declared operation order. Public operation vectors remain available for
/// inspection; slots retain their order when their payload is edited.
#[derive(Clone)]
pub(crate) enum PipelineStep {
    Image { index: usize, identity: Option<u64> },
    Index(usize),
    Batch,
    DeviceCut(usize),
}

#[derive(Clone)]
pub struct ImagePipeline {
    pub(crate) steps: Vec<PipelineStep>,
    pub source: SourceOp,
    pub index_ops: Vec<IndexOp>,
    pub ops: Vec<ImageOp>,
    pub batch: Option<BatchConfig>,
    pub(crate) device_cuts: Vec<DeviceCutPlacement>,
    pub(crate) batch_mixing: Vec<crate::batch::BatchMixConfig>,
    pub(crate) batch_mix_boundary: Option<(usize, usize, usize, bool)>,
    pub runtime: RuntimeConfig,
    pub epoch: u64,
    pub global_seed: Option<u64>,
}

impl ImagePipeline {
    fn push_image(&mut self, op: ImageOp) {
        self.steps.push(PipelineStep::Image {
            index: self.ops.len(),
            identity: None,
        });
        self.ops.push(op);
    }

    fn push_index(&mut self, op: IndexOp) {
        self.steps.push(PipelineStep::Index(self.index_ops.len()));
        self.index_ops.push(op);
    }

    pub fn new<T>(dataset: Arc<T>) -> Self
    where
        T: Dataset<Item = EncodedImageSample> + 'static,
    {
        Self::from_source(ImageSource::from_encoded(dataset))
    }

    pub fn from_source(source: ImageSource) -> Self {
        Self {
            steps: Vec::new(),
            source: SourceOp::new(source),
            index_ops: Vec::new(),
            ops: Vec::new(),
            batch: None,
            device_cuts: Vec::new(),
            batch_mixing: Vec::new(),
            batch_mix_boundary: None,
            runtime: RuntimeConfig::default(),
            epoch: 0,
            global_seed: None,
        }
    }

    /// Append terminal batch mixing after an explicit batch barrier.
    pub fn batch_mix(mut self, config: crate::batch::BatchMixConfig) -> Self {
        if self.batch_mix_boundary.is_none() {
            self.batch_mix_boundary = Some((
                self.steps.len(),
                self.ops.len(),
                self.index_ops.len(),
                self.batch.is_some(),
            ));
        }
        self.batch_mixing.push(config);
        self
    }

    /// Add a torchvision v2 image transform.
    pub fn v2(mut self, config: crate::transforms::V2Config) -> Self {
        self.push_image(ImageOp::V2(config));
        self
    }

    /// Add Gaussian noise with mean and standard deviation in normalized units.
    pub fn gaussian_noise(self, mean: f64, sigma: f64, clip: bool) -> Self {
        self.v2(crate::transforms::V2Config::GaussianNoise { mean, sigma, clip })
    }

    /// Add signal-dependent Gaussian noise to a normalized float32 image.
    pub fn letterbox(self, config: crate::transforms::LetterBoxConfig) -> Self {
        self.v2(crate::transforms::V2Config::LetterBox(config))
    }

    /// Add seeded offsets drawn from the requested distribution and sharing mode.
    pub fn additive_noise(
        self,
        distribution: crate::transforms::AdditiveDistribution,
        sharing: crate::transforms::NoiseSharing,
    ) -> Self {
        self.v2(crate::transforms::V2Config::AdditiveNoise(
            std::sync::Arc::new(crate::transforms::AdditiveNoiseConfig {
                distribution,
                sharing,
            }),
        ))
    }

    /// Apply seeded spatially and/or channel-shared multiplicative noise.
    pub fn multiplicative_noise(
        self,
        multiplier: (f64, f64),
        per_channel: bool,
        elementwise: bool,
    ) -> Self {
        self.v2(crate::transforms::V2Config::MultiplicativeNoise {
            multiplier,
            per_channel,
            elementwise,
        })
    }

    /// Render a seeded optical flare.
    pub fn random_sun_flare(self, config: crate::transforms::RandomSunFlareConfig) -> Self {
        self.v2(crate::transforms::V2Config::RandomSunFlare(
            std::sync::Arc::new(config),
        ))
    }

    /// Render a seeded optical flare.
    pub fn lens_flare(self, config: crate::transforms::LensFlareConfig) -> Self {
        self.v2(crate::transforms::V2Config::LensFlare(std::sync::Arc::new(
            config,
        )))
    }

    /// Apply seeded contrast and brightness in a fixed order.
    pub fn random_brightness_contrast(
        self,
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        brightness_by_max: bool,
        ensure_safe_output: bool,
    ) -> Self {
        self.v2(crate::transforms::V2Config::RandomBrightnessContrast {
            brightness_range,
            contrast_range,
            brightness_by_max,
            ensure_safe_output,
        })
    }

    /// Sample HSV shifts in degrees and 8-bit saturation/value units.
    pub fn hue_saturation_value(self, ranges: [(f64, f64); 3]) -> Self {
        self.v2(crate::transforms::V2Config::HueSaturationValue(
            std::sync::Arc::new(ranges),
        ))
    }

    /// Reorder channels using a complete permutation of the input indices.
    pub fn channel_swap(self, channel_order: Vec<usize>) -> Self {
        self.v2(crate::transforms::V2Config::PermuteChannels(channel_order))
    }

    /// Replace a seeded subset of eligible channels.
    pub fn channel_dropout(self, config: crate::transforms::ChannelDropoutConfig) -> Self {
        self.v2(crate::transforms::V2Config::ChannelDropout(
            std::sync::Arc::new(config),
        ))
    }

    /// Fill seeded horizontal and vertical strips.
    pub fn xy_masking(self, config: crate::transforms::XyMaskingConfig) -> Self {
        self.v2(crate::transforms::V2Config::XyMasking(std::sync::Arc::new(
            config,
        )))
    }

    /// Replace independently sampled pixels or channel elements.
    pub fn pixel_dropout(self, config: crate::transforms::PixelDropoutConfig) -> Self {
        self.v2(crate::transforms::V2Config::PixelDropout(
            std::sync::Arc::new(config),
        ))
    }

    /// Render luminance-dependent dots with seeded cell size and blend.
    pub fn halftone(self, dot_size_range: (u32, u32), blend_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::Halftone {
            dot_size_range,
            blend_range,
        })
    }

    /// Apply palette quantization with the configured dithering method.
    pub fn dithering(self, config: crate::transforms::DitheringConfig) -> Self {
        self.v2(crate::transforms::V2Config::Dithering(config))
    }

    /// Apply a radial sinc filter to simulate ringing at edges.
    pub fn ringing_overshoot(self, blur_range: (u32, u32), cutoff_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::RingingOvershoot {
            blur_range,
            cutoff_range,
        })
    }

    /// Apply the selected local detail or edge enhancement preset.
    pub fn enhance(self, mode: crate::transforms::EnhanceMode, alpha_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::Enhance { mode, alpha_range })
    }

    /// Blend a directional relief filter with the original image.
    pub fn emboss(self, alpha_range: (f64, f64), strength_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::Emboss {
            alpha_range,
            strength_range,
        })
    }

    /// Match each image to a seeded normalized target mean.
    pub fn exposure_matching(
        self,
        target_mean_range: (f64, f64),
        gain_range: Option<(f64, f64)>,
    ) -> Self {
        self.v2(crate::transforms::V2Config::ExposureMatching {
            target_mean_range,
            gain_range,
        })
    }

    /// Add seeded covariance-weighted principal component offsets.
    pub fn fancy_pca(self, alpha: f64) -> Self {
        self.v2(crate::transforms::V2Config::FancyPca { alpha })
    }

    /// Map grayscale intensities to a seeded RGB ramp.
    pub fn colorize(self, config: crate::transforms::ColorizeConfig) -> Self {
        self.v2(crate::transforms::V2Config::Colorize(config))
    }

    /// Apply a fixed sepia color matrix blended with the original RGB.
    pub fn to_sepia(self, alpha: f64) -> Self {
        self.v2(crate::transforms::V2Config::ToSepia { alpha })
    }

    /// Add independent seeded RGB offsets in 8-bit intensity units.
    pub fn rgb_shift(self, ranges: [(f64, f64); 3]) -> Self {
        self.v2(crate::transforms::V2Config::RgbShift(std::sync::Arc::new(
            ranges,
        )))
    }

    /// Apply a power-law exponent sampled from percent units.
    pub fn random_gamma(self, gamma_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::RandomGamma { gamma_range })
    }

    /// Apply seeded fog to an RGB image.
    pub fn random_fog(self, config: crate::transforms::RandomFogConfig) -> Self {
        self.v2(crate::transforms::V2Config::RandomFog(config))
    }

    /// Apply seeded fog to an RGB image.
    pub fn atmospheric_fog(self, config: crate::transforms::AtmosphericFogConfig) -> Self {
        self.v2(crate::transforms::V2Config::AtmosphericFog(config))
    }

    /// Overlay seeded rain or mud on the lens.
    pub fn spatter(self, config: crate::transforms::SpatterConfig) -> Self {
        self.v2(crate::transforms::V2Config::Spatter(std::sync::Arc::new(
            config,
        )))
    }

    /// Scatter seeded gravel particles in a normalized region.
    pub fn random_gravel(self, config: crate::transforms::RandomGravelConfig) -> Self {
        self.v2(crate::transforms::V2Config::RandomGravel(config))
    }

    /// Simulate snow using lightness bleaching or a seeded texture.
    pub fn random_snow(self, config: crate::transforms::RandomSnowConfig) -> Self {
        self.v2(crate::transforms::V2Config::RandomSnow(config))
    }

    /// Render seeded rain streaks and apply wet-scene blur and dimming.
    pub fn random_rain(self, config: crate::transforms::RandomRainConfig) -> Self {
        self.v2(crate::transforms::V2Config::RandomRain(
            std::sync::Arc::new(config),
        ))
    }

    /// Darken seeded polygonal regions of the image.
    pub fn random_shadow(self, config: crate::transforms::RandomShadowConfig) -> Self {
        self.v2(crate::transforms::V2Config::RandomShadow(
            std::sync::Arc::new(config),
        ))
    }

    /// Apply a seeded plasma light field.
    pub fn plasma_shadow(
        self,
        shadow_intensity_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    ) -> Self {
        self.v2(crate::transforms::V2Config::PlasmaShadow {
            shadow_intensity_range,
            plasma_size,
            roughness,
        })
    }

    /// Apply a seeded plasma light field.
    pub fn plasma_brightness_contrast(
        self,
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    ) -> Self {
        self.v2(crate::transforms::V2Config::PlasmaBrightnessContrast {
            brightness_range,
            contrast_range,
            plasma_size,
            roughness,
        })
    }

    /// Apply a smooth, seeded illumination field.
    pub fn illumination(
        self,
        intensity_range: (f64, f64),
        mode: crate::transforms::IlluminationMode,
        effect: crate::transforms::IlluminationEffect,
    ) -> Self {
        self.v2(crate::transforms::V2Config::Illumination {
            intensity_range,
            mode,
            effect,
        })
    }

    /// Overlay seeded, spatially correlated grain with stronger shadows.
    pub fn film_grain(
        self,
        intensity_range: (f64, f64),
        grain_size_range: (u32, u32),
        per_channel: bool,
    ) -> Self {
        self.v2(crate::transforms::V2Config::FilmGrain {
            intensity_range,
            grain_size_range,
            per_channel,
        })
    }

    /// Apply camera-like luma and chroma noise to RGB pixels.
    pub fn iso_noise(self, color_shift_range: (f64, f64), intensity_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::IsoNoise {
            color_shift_range,
            intensity_range,
        })
    }

    /// Apply Gaussian smoothing around seeded local pixel exchanges.
    pub fn glass_blur(self, sigma: f64, max_delta: u32, iterations: u32) -> Self {
        self.v2(crate::transforms::V2Config::GlassBlur {
            sigma,
            max_delta,
            iterations,
        })
    }

    /// Average an image with centered, bilinearly zoomed samples.
    pub fn zoom_blur(self, max_factor_range: (f64, f64), step_factor_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::ZoomBlur {
            max_factor_range,
            step_factor_range,
        })
    }

    /// Apply a seeded Gaussian perturbation of the identity kernel.
    pub fn stochastic_convolution(
        self,
        kernel_range: (u32, u32),
        strength_range: (f32, f32),
        per_channel: bool,
        padding_mode: crate::transforms::PaddingMode,
    ) -> Self {
        self.v2(crate::transforms::V2Config::StochasticConvolution {
            kernel_range,
            strength_range,
            per_channel,
            padding_mode,
        })
    }

    /// Apply a seeded, rotated generalized Gaussian blur.
    #[allow(clippy::too_many_arguments)]
    pub fn advanced_blur(
        self,
        kernel_range: (u32, u32),
        sigma_x_range: (f64, f64),
        sigma_y_range: (f64, f64),
        rotate_range: (f64, f64),
        beta_range: (f64, f64),
        noise_range: (f64, f64),
    ) -> Self {
        self.v2(crate::transforms::V2Config::AdvancedBlur(
            std::sync::Arc::new(crate::transforms::AdvancedBlurConfig {
                kernel_range,
                sigma_x_range,
                sigma_y_range,
                rotate_range,
                beta_range,
                noise_range,
            }),
        ))
    }

    /// Apply a box filter with a seeded odd kernel width.
    pub fn blur(self, kernel_range: (u32, u32)) -> Self {
        self.v2(crate::transforms::V2Config::Blur { kernel_range })
    }

    /// Apply a channel-wise 3×3 median filter with replicated border pixels.
    pub fn median_blur(self) -> Self {
        self.v2(crate::transforms::V2Config::MedianBlur)
    }

    /// Sample a blackbody temperature in kelvin to tint an RGB image.
    pub fn planckian_jitter(self, temperature_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::PlanckianJitter { temperature_range })
    }

    /// Apply seeded radial darkening with intensity and center fractions in [0, 1].
    pub fn vignetting(self, intensity_range: (f64, f64), center_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::Vignetting {
            intensity_range,
            center_range,
        })
    }

    pub fn random_tone_curve(self, scale: f64) -> Self {
        self.v2(crate::transforms::V2Config::RandomToneCurve { scale })
    }

    pub fn clahe(self, grid: (u32, u32), clip_limit: f64) -> Self {
        self.v2(crate::transforms::V2Config::Clahe { grid, clip_limit })
    }

    pub fn chromatic_aberration(
        self,
        red_range: (f64, f64),
        blue_range: (f64, f64),
        interpolation: InterpolationMode,
    ) -> Self {
        self.v2(crate::transforms::V2Config::ChromaticAberration {
            red_range,
            blue_range,
            interpolation,
        })
    }

    pub fn defocus(self, radius_range: (u32, u32)) -> Self {
        self.v2(crate::transforms::V2Config::Defocus { radius_range })
    }

    pub fn coarse_dropout(
        self,
        holes: (u32, u32),
        hole_height: (u32, u32),
        hole_width: (u32, u32),
        fill: f32,
    ) -> Self {
        self.v2(crate::transforms::V2Config::CoarseDropout {
            holes,
            hole_height,
            hole_width,
            fill,
        })
    }

    pub fn motion_blur(self, length_range: (u32, u32), angle_range: (f64, f64)) -> Self {
        self.v2(crate::transforms::V2Config::MotionBlur {
            length_range,
            angle_range,
        })
    }

    pub fn unsharp_mask(self, sigma: f32, amount: f32, threshold: u8) -> Self {
        self.v2(crate::transforms::V2Config::UnsharpMask {
            sigma,
            amount,
            threshold,
        })
    }

    pub fn downscale(
        self,
        scale_range: (f64, f64),
        down_interpolation: InterpolationMode,
        up_interpolation: InterpolationMode,
    ) -> Self {
        self.v2(crate::transforms::V2Config::Downscale {
            scale_range,
            down_interpolation,
            up_interpolation,
        })
    }

    pub fn salt_and_pepper(self, amount: f64, salt_ratio: f64) -> Self {
        self.v2(crate::transforms::V2Config::SaltAndPepper { amount, salt_ratio })
    }

    pub fn grid_dropout(self, unit_size: usize, ratio: f64, fill: f32) -> Self {
        self.v2(crate::transforms::V2Config::GridDropout {
            unit_size,
            ratio,
            fill,
        })
    }

    pub fn shot_noise(self, intensity: f32) -> Self {
        self.v2(crate::transforms::V2Config::ShotNoise { intensity })
    }

    pub fn auto_augment(mut self, config: crate::transforms::AutoAugmentConfig) -> Self {
        self.push_image(ImageOp::AutoAugment(config));
        self
    }
    pub fn rand_augment(mut self, config: crate::transforms::RandAugmentConfig) -> Self {
        self.push_image(ImageOp::RandAugment(config));
        self
    }
    pub fn trivial_augment_wide(
        mut self,
        config: crate::transforms::TrivialAugmentWideConfig,
    ) -> Self {
        self.push_image(ImageOp::TrivialAugmentWide(config));
        self
    }
    pub fn augmix(mut self, config: crate::transforms::AugMixConfig) -> Self {
        self.push_image(ImageOp::AugMix(config));
        self
    }
    pub fn linear_transform(
        mut self,
        config: crate::transforms::LinearTransformationConfig,
    ) -> Self {
        self.push_image(ImageOp::LinearTransformation(config));
        self
    }
    pub fn to_tensor(self) -> Self {
        self.convert_image_dtype(DType::F32).hwc_to_chw()
    }
    pub fn pil_to_tensor(self) -> Self {
        self.hwc_to_chw()
    }
    pub fn saturation(mut self, factor: f32) -> Self {
        self.push_image(ImageOp::Saturation(
            crate::transforms::SaturationConfig::new(factor),
        ));
        self
    }
    pub fn gamma(mut self, gamma: f64, gain: f64) -> Self {
        self.push_image(ImageOp::Gamma(crate::transforms::GammaConfig::new(
            gamma, gain,
        )));
        self
    }
    pub fn affine(mut self, config: crate::transforms::AffineConfig) -> Self {
        self.push_image(ImageOp::Affine(config));
        self
    }
    pub fn erase(mut self, config: crate::transforms::EraseConfig) -> Self {
        self.push_image(ImageOp::Erase(config));
        self
    }
    pub fn random_rotation(mut self, config: crate::transforms::RandomRotationConfig) -> Self {
        self.push_image(ImageOp::RandomRotation(config));
        self
    }
    pub fn random_gaussian_blur(
        mut self,
        config: crate::transforms::RandomGaussianBlurConfig,
    ) -> Self {
        self.push_image(ImageOp::RandomGaussianBlur(config));
        self
    }
    pub fn color_jitter_with_config(
        mut self,
        config: crate::transforms::ColorJitterConfig,
    ) -> Self {
        self.push_image(ImageOp::ColorJitter(config));
        self
    }
    pub fn resized_crop(
        self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        out_width: u32,
        out_height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        self.crop(x, y, width, height).resize_with_interpolation(
            out_width,
            out_height,
            interpolation,
        )
    }
    pub fn random_vertical_flip(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(
            probability,
            vec![ImageOp::vertical_flip()],
        ));
        self
    }
    pub fn random_invert(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(probability, vec![ImageOp::invert()]));
        self
    }
    pub fn random_posterize(mut self, bits: u8, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(
            probability,
            vec![ImageOp::posterize(bits)],
        ));
        self
    }
    pub fn random_solarize(mut self, threshold: u8, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(
            probability,
            vec![ImageOp::solarize(threshold)],
        ));
        self
    }
    pub fn random_adjust_sharpness(mut self, factor: f32, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(
            probability,
            vec![ImageOp::sharpness(factor)],
        ));
        self
    }
    pub fn random_autocontrast(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(
            probability,
            vec![ImageOp::autocontrast()],
        ));
        self
    }
    pub fn random_equalize(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_apply(
            probability,
            vec![ImageOp::equalize()],
        ));
        self
    }

    pub fn decode_image(mut self) -> Self {
        self.push_image(ImageOp::decode());
        self
    }

    pub fn resize(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::resize(width, height));
        self
    }

    pub fn resize_short_edge(mut self, size: u32, max_size: Option<u32>) -> Self {
        self.push_image(ImageOp::resize_short_edge(
            size,
            max_size,
            crate::transforms::InterpolationMode::Bilinear,
        ));
        self
    }

    pub fn resize_short_edge_with_interpolation(
        mut self,
        size: u32,
        max_size: Option<u32>,
        interpolation: crate::transforms::InterpolationMode,
    ) -> Self {
        self.push_image(ImageOp::resize_short_edge(size, max_size, interpolation));
        self
    }

    pub fn resize_with_interpolation(
        mut self,
        width: u32,
        height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        self.push_image(ImageOp::resize_with_interpolation(
            width,
            height,
            interpolation,
        ));
        self
    }

    /// Resize proportionally to the longest side or explicit axis limits.
    pub fn longest_max_size(self, config: crate::transforms::LongestMaxSizeConfig) -> Self {
        self.v2(crate::transforms::V2Config::LongestMaxSize(config))
    }

    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::crop(x, y, width, height));
        self
    }

    /// Expand each source image into the four corner crops and center crop.
    pub fn five_crop(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::five_crop(width, height));
        self
    }

    /// Five crops of the original image followed by five of its flipped image.
    pub fn ten_crop(mut self, width: u32, height: u32, vertical_flip: bool) -> Self {
        self.push_image(ImageOp::ten_crop(width, height, vertical_flip));
        self
    }

    pub fn center_crop(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::center_crop(width, height));
        self
    }

    pub fn pad_with_config(mut self, config: crate::transforms::PadConfig) -> Self {
        self.push_image(ImageOp::Pad(config));
        self
    }

    pub fn pad(mut self, padding: u32) -> Self {
        self.push_image(ImageOp::pad(padding));
        self
    }

    pub fn pad_with_fill(mut self, padding: u32, value: f32) -> Self {
        self.push_image(ImageOp::pad_with_fill(padding, value));
        self
    }

    pub fn pad_with_sides(
        mut self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        value: f32,
    ) -> Self {
        self.push_image(ImageOp::pad_with_sides(left, top, right, bottom, value));
        self
    }

    pub fn horizontal_flip(mut self) -> Self {
        self.push_image(ImageOp::horizontal_flip());
        self
    }

    pub fn vertical_flip(mut self) -> Self {
        self.push_image(ImageOp::vertical_flip());
        self
    }

    /// Zero-pad every decoded image by `padding` pixels on each side, then
    /// crop a `width`x`height` window at a per-sample random offset.
    ///
    /// Draws come from the pipeline's stochastic seed (the shuffle seed
    /// when the pipeline shuffles), so the same `(seed, epoch)` reproduces
    /// the same augmentations at any worker count.
    pub fn random_crop(mut self, width: u32, height: u32, padding: u32) -> Self {
        self.push_image(ImageOp::random_crop(width, height, padding));
        self
    }

    pub fn random_crop_with_config(mut self, config: crate::transforms::RandomCropConfig) -> Self {
        self.push_image(ImageOp::random_crop_with_config(config));
        self
    }

    pub fn random_resized_crop(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::random_resized_crop(width, height));
        self
    }

    pub fn random_resized_crop_with_config(mut self, config: RandomResizedCropConfig) -> Self {
        self.push_image(ImageOp::random_resized_crop_with_config(config));
        self
    }

    /// Randomly flip each decoded image horizontally with `probability`,
    /// per sample, from the pipeline's stochastic seed.
    pub fn random_horizontal_flip(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_horizontal_flip(probability));
        self
    }

    pub fn brightness(mut self, value: f32) -> Self {
        self.push_image(ImageOp::brightness(value));
        self
    }

    pub fn contrast(mut self, value: f32) -> Self {
        self.push_image(ImageOp::contrast(value));
        self
    }

    pub fn color_jitter(mut self, brightness: f32, contrast: f32, hue: f32) -> Self {
        self.push_image(ImageOp::color_jitter(brightness, contrast, hue));
        self
    }

    pub fn invert(mut self) -> Self {
        self.push_image(ImageOp::invert());
        self
    }

    pub fn posterize(mut self, bits: u8) -> Self {
        self.push_image(ImageOp::posterize(bits));
        self
    }

    pub fn solarize(mut self, threshold: u8) -> Self {
        self.push_image(ImageOp::solarize(threshold));
        self
    }

    pub fn autocontrast(mut self) -> Self {
        self.push_image(ImageOp::autocontrast());
        self
    }

    pub fn equalize(mut self) -> Self {
        self.push_image(ImageOp::equalize());
        self
    }

    pub fn sharpness(mut self, amount: f32) -> Self {
        self.push_image(ImageOp::sharpness(amount));
        self
    }

    pub fn gaussian_blur_with_config(
        mut self,
        config: crate::transforms::GaussianBlurConfig,
    ) -> Self {
        self.push_image(ImageOp::GaussianBlur(config));
        self
    }

    pub fn gaussian_blur(mut self, sigma: f32) -> Self {
        self.push_image(ImageOp::gaussian_blur(sigma));
        self
    }

    pub fn hue(mut self, factor: f32) -> Self {
        self.push_image(ImageOp::hue(factor));
        self
    }

    pub fn grayscale(mut self, num_output_channels: u8) -> Self {
        self.push_image(ImageOp::grayscale(num_output_channels));
        self
    }

    pub fn random_grayscale(mut self, probability: f64, num_output_channels: u8) -> Self {
        self.push_image(ImageOp::random_grayscale(probability, num_output_channels));
        self
    }

    pub fn random_erasing(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_erasing(probability));
        self
    }

    pub fn random_erasing_with_config(mut self, config: RandomErasingConfig) -> Self {
        self.push_image(ImageOp::random_erasing_with_config(config));
        self
    }

    pub fn convert_image_dtype(mut self, dtype: DType) -> Self {
        self.push_image(ImageOp::convert_image_dtype(dtype));
        self
    }

    pub fn rotate(mut self, angle: crate::transforms::RotationAngle) -> Self {
        self.push_image(ImageOp::rotate(angle));
        self
    }

    pub fn arbitrary_rotate(mut self, angle: f32) -> Self {
        self.push_image(ImageOp::arbitrary_rotate(angle));
        self
    }

    pub fn arbitrary_rotate_with_options(
        mut self,
        angle: f32,
        expand: bool,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> Self {
        self.push_image(ImageOp::arbitrary_rotate_with_options(
            angle,
            expand,
            interpolation,
            fill,
        ));
        self
    }

    pub fn arbitrary_rotate_with_config(mut self, config: ArbitraryRotateConfig) -> Self {
        self.push_image(ImageOp::arbitrary_rotate_with_config(config));
        self
    }

    pub fn random_affine(mut self, degrees: f32) -> Self {
        self.push_image(ImageOp::random_affine(degrees));
        self
    }

    pub fn random_affine_with_config(mut self, config: RandomAffineConfig) -> Self {
        self.push_image(ImageOp::random_affine_with_config(config));
        self
    }

    pub fn perspective(mut self, start_points: [Point2; 4], end_points: [Point2; 4]) -> Self {
        self.push_image(ImageOp::perspective(start_points, end_points));
        self
    }

    pub fn perspective_with_config(mut self, config: PerspectiveConfig) -> Self {
        self.push_image(ImageOp::perspective_with_config(config));
        self
    }

    pub fn random_perspective(mut self, distortion_scale: f32, probability: f64) -> Self {
        self.push_image(ImageOp::random_perspective(distortion_scale, probability));
        self
    }

    pub fn random_perspective_with_config(mut self, config: RandomPerspectiveConfig) -> Self {
        self.push_image(ImageOp::random_perspective_with_config(config));
        self
    }

    pub fn elastic_transform(mut self, alpha: f32, sigma: f32) -> Self {
        self.push_image(ImageOp::elastic_transform(alpha, sigma));
        self
    }

    pub fn elastic_transform_with_config(mut self, config: ElasticTransformConfig) -> Self {
        self.push_image(ImageOp::elastic_transform_with_config(config));
        self
    }

    pub fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.push_image(ImageOp::normalize(mean, std));
        self
    }

    pub fn hwc_to_chw(mut self) -> Self {
        self.push_image(ImageOp::hwc_to_chw());
        self
    }

    pub fn chw_to_hwc(mut self) -> Self {
        self.push_image(ImageOp::chw_to_hwc());
        self
    }

    /// Append a reusable transform sequence to this pipeline.
    pub fn compose(mut self, sequence: TransformSequence) -> Self {
        for op in sequence.into_ops() {
            self.push_image(op);
        }
        self
    }

    pub fn random_apply(mut self, probability: f64, sequence: TransformSequence) -> Self {
        self.push_image(ImageOp::random_apply(probability, sequence.into_ops()));
        self
    }

    pub fn random_choice(mut self, choices: Vec<TransformSequence>) -> Self {
        self.push_image(ImageOp::random_choice(
            choices
                .into_iter()
                .map(TransformSequence::into_ops)
                .collect(),
        ));
        self
    }

    /// Choose a transform sequence using nonnegative relative weights.
    pub fn random_choice_with_weights(
        mut self,
        choices: Vec<TransformSequence>,
        weights: Vec<f64>,
    ) -> Self {
        self.push_image(ImageOp::random_choice_with_weights(
            choices
                .into_iter()
                .map(TransformSequence::into_ops)
                .collect(),
            weights,
        ));
        self
    }

    pub fn random_order(mut self, sequence: TransformSequence) -> Self {
        self.push_image(ImageOp::random_order(sequence.into_ops()));
        self
    }

    pub fn skip(mut self, count: usize) -> Self {
        self.push_index(IndexOp::Skip { count });
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        self.push_index(IndexOp::Take { count });
        self
    }

    /// Deterministically shuffle the sampled window with `seed`; the same
    /// semantic seed and epoch reproduce the same order at any worker count.
    /// Prefer `.seed(seed)` when sampler and transform randomness should share
    /// an explicit pipeline-owned namespace.
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.push_index(IndexOp::Shuffle { seed });
        self
    }

    /// Set the pipeline-owned semantic seed used by sampling and transforms.
    /// A legacy `.shuffle(seed)` remains a fallback when this is omitted.
    pub fn seed(mut self, seed: u64) -> Self {
        self.global_seed = Some(seed);
        self
    }

    /// Set the explicit epoch used by both sampler and transform namespaces.
    /// Use the same epoch with different worker counts to reproduce exactly;
    /// changing it produces a fresh permutation and per-sample RNG stream.
    pub fn epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    pub fn batch(mut self, size: usize, drop_last: bool) -> Self {
        let config = BatchConfig::new(size, drop_last);
        self.steps.push(PipelineStep::Batch);
        self.batch = Some(config);
        self
    }

    /// Insert a fixed host-to-Metal semantic boundary after the image
    /// transformations appended so far. A cut follows the batch node when
    /// batching is already configured. Runtime execution requires a Metal
    /// backend implementation.
    pub fn device_cut_to_metal(self, ordinal: usize) -> Self {
        self.device_cut(DeviceTarget::metal(ordinal))
    }

    /// Insert a fixed host-to-device semantic boundary after the image
    /// transformations appended so far. If batching is already configured,
    /// the cut follows the batch node; otherwise it precedes batching.
    pub fn device_cut(mut self, target: DeviceTarget) -> Self {
        self.steps
            .push(PipelineStep::DeviceCut(self.device_cuts.len()));
        self.device_cuts.push(DeviceCutPlacement {
            after_ops: self.ops.len(),
            after_batch: self.batch.is_some(),
            target,
        });
        self
    }

    /// Execute per-sample semantic transforms on a persistent pool of
    /// `num_workers` inter-sample workers (`0` executes them on the CPU
    /// transform stage thread). Decode runs on its own persistent stage;
    /// backend-internal parallelism remains controlled by each kernel/backend.
    /// Ordering, batching and sampling semantics do not depend on worker count.
    pub fn workers(mut self, num_workers: usize) -> Self {
        self.runtime.num_workers = num_workers;
        self
    }

    /// Allow up to `prefetch_batches` future batches behind the current
    /// delivery (`prefetch_batches + 1` outstanding sequences total). Stage
    /// queues apply the configured item and byte bounds; delivery stays in
    /// sampler order.
    pub fn prefetch_batches(mut self, prefetch_batches: usize) -> Self {
        self.runtime.prefetch_batches = prefetch_batches;
        self
    }

    /// Set the DAG single-output reported logical byte limit independently of
    /// persistent queue weights. Checked after kernel allocation, not admission.
    pub fn dag_max_value_logical_bytes(mut self, bytes: usize) -> Self {
        self.runtime.dag_max_value_logical_bytes = Some(bytes);
        self
    }

    /// Bound the producer-reported payload weight on each persistent queue.
    /// Defaults to 512 MiB; dequeued/active values are excluded and the CPU pull
    /// path has no queues. Legacy DAG fallback supplies its single-value limit
    /// unless dag_max_value_logical_bytes was explicitly set. Not a total budget.
    pub fn stage_queue_max_bytes(mut self, max_bytes: usize) -> Self {
        self.runtime.stage_queue_max_bytes = max_bytes;
        self
    }

    /// Enable detailed per-node profiling for diagnostics.
    /// Profiling is disabled by default to avoid steady-state timing and
    /// counter updates on the runtime hot path.
    pub fn profiling(mut self, enabled: bool) -> Self {
        self.runtime.profiling_enabled = enabled;
        self
    }
}
