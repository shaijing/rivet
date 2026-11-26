use super::V2Config;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::representation::channels::{convert_rgb, permute};
use crate::transforms::{
    AdvancedBlurConfig, BrightnessConfig, ContrastConfig, ConvertImageDtypeConfig, CropConfig,
    HueConfig, InterpolationMode, LayoutConfig, PadConfig, ResizeConfig, SaturationConfig,
    SquareSymmetry, codecs, degradation, filtering, gaussian_noise, occlusion, photometric,
    quarter_turn, representation, require_u8_hwc, sample_border_crop, sample_scale, shot_noise,
    spatial,
};
use rivet_core::{DType, Tensor};
use rivet_data::random::RandomStream;

impl V2Config {
    pub fn apply_batch(&self, input: Tensor, axis: ImageAxisOrder) -> RivetResult<Tensor> {
        self.validate()?;
        if input.rank() != 4 {
            return Err(invalid_shape("batch ToDtype requires rank-4 image input"));
        }
        if let Self::ToDtype { dtype, scale } = self {
            if *scale {
                ConvertImageDtypeConfig::new(*dtype).apply_batch(input, axis)
            } else {
                representation::numeric::cast_unscaled(&input, *dtype)
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
                s.image = representation::numeric::cast_unscaled(&s.image, *dtype)?;
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
                let [x, y, w, h] = sample_border_crop(h, w, *fractions, rng)?;
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
                s.image = gaussian_noise(&s.image, *mean, *sigma, *clip, rng)?;
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
            Self::HEStain(config) => {
                let input = match axis {
                    ImageAxisOrder::Hwc => s.image,
                    ImageAxisOrder::Chw => s.image.permute(&[1, 2, 0])?,
                };
                let image = config.apply(&input, rng)?;
                s.image = match axis {
                    ImageAxisOrder::Hwc => image,
                    ImageAxisOrder::Chw => image.permute(&[2, 0, 1])?,
                };
            }
            Self::KSpaceSpikeNoise(config) => {
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
                let image =
                    degradation::noise::rician_noise(&input, c, *std_range, *per_channel, rng)?;
                s.image = match axis {
                    ImageAxisOrder::Hwc => image,
                    ImageAxisOrder::Chw => image.permute(&[2, 0, 1])?,
                };
            }
            Self::ShotNoise { intensity } => {
                s.image = shot_noise(&s.image, *intensity, rng)?;
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
                s.image = filtering::blur::box_blur(&s.image, size as usize)?;
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
                s.image = filtering::blur::advanced_blur(
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
                s.image = filtering::blur::stochastic_convolution(
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
                s.image = filtering::blur::zoom_blur(&s.image, max_factor, step)?;
            }
            Self::GlassBlur {
                sigma,
                max_delta,
                iterations,
            } => {
                s.image = filtering::blur::glass_blur(
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
                s.image = degradation::noise::iso_noise(&s.image, color_shift, intensity, rng)?;
            }
            Self::FilmGrain {
                intensity_range: (low, high),
                grain_size_range: (min, max),
                per_channel,
            } => {
                let intensity = low + rng.next_f64() * (high - low);
                let size = *min as usize + rng.gen_range_usize(0..((*max - *min) as usize + 1))?;
                s.image =
                    degradation::noise::film_grain(&s.image, intensity, size, *per_channel, rng)?;
            }
            Self::MultiplicativeNoise {
                multiplier,
                per_channel,
                elementwise,
            } => {
                s.image = degradation::noise::multiplicative_noise(
                    &s.image,
                    *multiplier,
                    *per_channel,
                    *elementwise,
                    rng,
                )?;
            }
            Self::AdditiveNoise(config) => {
                s.image = degradation::noise::additive_noise(
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
                s.image = photometric::local_color::illumination(
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
                s.image = photometric::local_color::apply_plasma(
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
                s.image = photometric::local_color::apply_plasma(
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
                s.image = photometric::local_color::random_shadow(&s.image, config, rng)?;
            }
            Self::RandomSunFlare(config) => {
                s.image = degradation::flare::random_sun_flare(&s.image, config, rng)?;
            }
            Self::LensFlare(config) => {
                s.image = degradation::flare::lens_flare(&s.image, config, rng)?;
            }
            Self::RandomRain(config) => {
                s.image = degradation::weather::random_rain(&s.image, config, rng)?;
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
                s.image = photometric::color::random_brightness_contrast(
                    &s.image,
                    brightness,
                    contrast,
                    *brightness_by_max,
                    *ensure_safe_output,
                )?;
            }
            Self::HueSaturationValue(ranges) => {
                let shifts = ranges.map(|(lo, hi)| lo + rng.next_f64() * (hi - lo));
                s.image = photometric::color::hue_saturation_value(&s.image, shifts)?;
            }
            Self::ExposureMatching {
                target_mean_range: (lo, hi),
                gain_range,
            } => {
                let target = lo + rng.next_f64() * (hi - lo);
                s.image = photometric::color::exposure_matching(&s.image, target, *gain_range)?;
            }
            Self::Emboss {
                alpha_range,
                strength_range,
            } => {
                let mut sample = |(lo, hi)| lo + rng.next_f64() * (hi - lo);
                s.image = filtering::blur::emboss(
                    &s.image,
                    sample(*alpha_range),
                    sample(*strength_range),
                )?;
            }
            Self::Enhance {
                mode,
                alpha_range: (lo, hi),
            } => {
                let alpha = lo + rng.next_f64() * (hi - lo);
                s.image = filtering::blur::enhance(&s.image, *mode, alpha)?;
            }
            Self::RingingOvershoot {
                blur_range: (lo, hi),
                cutoff_range: (min, max),
            } => {
                let size =
                    *lo as usize + 2 * rng.gen_range_usize(0..((hi - lo) / 2 + 1) as usize)?;
                let cutoff = min + rng.next_f64() * (max - min);
                s.image = filtering::blur::ringing_overshoot(&s.image, size, cutoff)?;
            }
            Self::Halftone {
                dot_size_range: (lo, hi),
                blend_range: (min, max),
            } => {
                let size = *lo as usize + rng.gen_range_usize(0..(*hi - *lo) as usize + 1)?;
                let blend = min + rng.next_f64() * (max - min);
                s.image = filtering::dither::halftone(&s.image, size, blend)?;
            }
            Self::Dithering(config) => {
                s.image = config.apply(&s.image, rng)?;
            }
            Self::FancyPca { alpha } => {
                s.image = photometric::pca::fancy_pca(&s.image, *alpha, rng)?;
            }
            Self::Colorize(config) => {
                s.image = config.apply(&s.image, rng)?;
            }
            Self::ToSepia { alpha } => {
                s.image = photometric::color::to_sepia(&s.image, *alpha)?;
            }
            Self::RgbShift(ranges) => {
                s.image = degradation::noise::rgb_shift(&s.image, ranges, rng)?;
            }
            Self::RandomGamma {
                gamma_range: (lo, hi),
            } => {
                let exponent = (lo + rng.next_f64() * (hi - lo)) / 100.0;
                s.image = photometric::color::random_gamma(&s.image, exponent)?;
            }
            Self::RandomFog(config) => {
                s.image = degradation::fog::random_fog(&s.image, config, rng)?;
            }
            Self::AtmosphericFog(config) => {
                s.image = degradation::fog::atmospheric_fog(&s.image, config, rng)?;
            }
            Self::Spatter(config) => {
                s.image = degradation::weather::spatter(&s.image, config, rng)?;
            }
            Self::RandomGravel(config) => {
                s.image = degradation::weather::random_gravel(&s.image, config, rng)?;
            }
            Self::RandomSnow(config) => {
                s.image = degradation::weather::random_snow(&s.image, config, rng)?;
            }
            Self::MedianBlur => {
                s.image = filtering::blur::median_blur_3x3(&s.image)?;
            }
            Self::RandomToneCurve { scale } => {
                s = require_u8_hwc(s, self.name())?;
                s.image = photometric::local_color::tone_curve(&s.image, *scale, rng)?;
            }
            Self::Vignetting {
                intensity_range,
                center_range,
            } => {
                let mut sample = |(low, high)| low + rng.next_f64() * (high - low);
                let intensity = sample(*intensity_range);
                let center = (sample(*center_range), sample(*center_range));
                s.image = photometric::local_color::vignette(&s.image, intensity, center)?;
            }
            Self::PlanckianJitter {
                temperature_range: (low, high),
            } => {
                let kelvin = low + rng.next_f64() * (high - low);
                s.image = photometric::local_color::planckian_jitter(&s.image, kelvin)?;
            }
            Self::Clahe { grid, clip_limit } => {
                s.image = photometric::local_color::clahe(&s.image, *grid, *clip_limit)?;
            }
            Self::ChromaticAberration {
                red_range,
                blue_range,
                interpolation,
            } => {
                let mut coefficient = |(low, high)| low + rng.next_f64() * (high - low);
                let red = coefficient(*red_range);
                let blue = coefficient(*blue_range);
                s.image = spatial::advanced_geometry::chromatic_aberration(
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
                s.image = filtering::blur::defocus(&s.image, radius)?;
            }
            Self::CoarseDropout {
                holes,
                hole_height,
                hole_width,
                fill,
            } => {
                s.image = occlusion::pixel_mask::coarse_dropout(
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
                s.image = filtering::blur::motion_blur(&s.image, length, angle)?;
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
                s.image = filtering::blur::unsharp_mask(s.clone(), *sigma, *amount, *threshold)?;
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
                s.image = occlusion::pixel_mask::salt_and_pepper(
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
                s.image = quarter_turn(&s.image, axis, turns)?;
            }
            Self::AnnotationArtifacts(config) => {
                if axis != ImageAxisOrder::Hwc {
                    return Err(invalid_argument("AnnotationArtifacts requires HWC input"));
                }
                s.image = config.apply(&s.image, rng)?;
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "WaterRefraction requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng)?;
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "PixelSpread requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng)?;
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "ThinPlateSpline requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "GridElasticDeform requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "PiecewiseAffine requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng)?;
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "GridDistortion requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                use spatial::remap::SpatialRemap;
                if s.image.dtype() != DType::U8 || h == 0 || w == 0 {
                    return Err(invalid_argument(
                        "OpticalDistortion requires nonempty uint8 input",
                    ));
                }
                let map = config.sample(h, w, rng);
                if !map.is_identity() {
                    return spatial::remap::remap_image(
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
                s.image = quarter_turn(&s.image, axis, turns)?;
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
                s.image = occlusion::pixel_mask::grid_dropout(
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
                let (w, h) = sample_scale(w, h, *range, *y, rng)?;
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
            Self::Webp {
                quality: (low, high),
            } => {
                let quality = rng.gen_range_usize(*low as usize..*high as usize + 1)? as u8;
                s.image = codecs::webp::webp(&s.image, axis, quality)?;
            }
            Self::Jpeg {
                quality: (low, high),
            } => {
                let q = rng.gen_range_usize(*low as usize..*high as usize + 1)? as u8;
                s.image = codecs::jpeg::jpeg(&s.image, axis, q)?;
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
