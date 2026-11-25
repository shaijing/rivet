use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::with_rgb_pixels;
use rivet_core::{DType, Device, Tensor};
use rivet_data::random::RandomStream;

pub use super::u8_color::{
    AutocontrastConfig, EqualizeConfig, InvertConfig, PosterizeConfig, SharpnessConfig,
    SolarizeConfig,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrightnessConfig {
    /// torchvision brightness factor: 0 is black, 1 is identity.
    pub value: f32,
}

impl BrightnessConfig {
    pub const fn new(value: f32) -> Self {
        Self { value }
    }
    pub fn validate(&self) -> RivetResult<()> {
        validate_factor(self.value, "brightness")
    }
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let lut: [u8; 256] = std::array::from_fn(|v| blend(0, v as u8, self.value));
        map_rgb_bytes(sample, "brightness", |value| lut[value as usize])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastConfig {
    pub value: f32,
}

impl ContrastConfig {
    pub const fn new(value: f32) -> Self {
        Self { value }
    }
    pub fn validate(&self) -> RivetResult<()> {
        validate_factor(self.value, "contrast")
    }
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let image = with_rgb_pixels(&sample.image, "contrast", |pixels, width, height| {
            // Compute the mean from the same rounded Pillow L values.
            let total: u64 = pixels.chunks_exact(3).map(|p| u64::from(luma(p))).sum();
            let count = width as u64 * height as u64;
            let mean = if count == 0 {
                0
            } else {
                (total as f64 / count as f64 + 0.5) as u8
            };
            let lut: [u8; 256] = std::array::from_fn(|v| blend(mean, v as u8, self.value));
            Ok(Tensor::from_exact_iter(
                pixels.iter().map(|&v| lut[v as usize]),
                [height, width, 3],
                sample.image.device(),
            )?)
        })?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HueConfig {
    /// torchvision hue offset in [-0.5, 0.5].
    pub factor: f32,
}

impl HueConfig {
    pub const fn new(factor: f32) -> Self {
        Self { factor }
    }
    pub fn validate(&self) -> RivetResult<()> {
        if self.factor.is_finite() && (-0.5..=0.5).contains(&self.factor) {
            Ok(())
        } else {
            Err(invalid_argument("hue factor must be in [-0.5, 0.5]"))
        }
    }
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let shift = (f64::from(self.factor) * 255.0) as i32 as u8;
        map_rgb(sample, "hue", |rgb| shift_hue(rgb, shift))
    }
}

/// Pillow ImageEnhance.Color: blend RGB against its rounded BT.601 luma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaturationConfig {
    pub factor: f32,
}
impl SaturationConfig {
    pub const fn new(factor: f32) -> Self {
        Self { factor }
    }
    pub fn validate(&self) -> RivetResult<()> {
        validate_factor(self.factor, "saturation")
    }
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        map_rgb(sample, "saturation", |rgb| {
            let gray = luma(&rgb);
            rgb.map(|v| blend(gray, v, self.factor))
        })
    }
}

/// torchvision's PIL gamma lookup table; exponentiation is done in f64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaConfig {
    pub gamma: f64,
    pub gain: f64,
}
impl GammaConfig {
    pub const fn new(gamma: f64, gain: f64) -> Self {
        Self { gamma, gain }
    }
    pub fn validate(&self) -> RivetResult<()> {
        if !self.gamma.is_finite() || self.gamma < 0.0 || !self.gain.is_finite() {
            return Err(invalid_argument(
                "gamma must be finite and non-negative; gain must be finite",
            ));
        }
        Ok(())
    }
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let lut = self.lut();
        map_rgb_bytes(sample, "gamma", |v| lut[v as usize])
    }

    fn lut(&self) -> [u8; 256] {
        std::array::from_fn(|v| {
            ((256.0 - 1e-3) * self.gain * (v as f64 / 255.0).powf(self.gamma)).clamp(0.0, 255.0)
                as u8
        })
    }
}

fn validate_factor(value: f32, name: &str) -> RivetResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "{name} factor must be finite and non-negative"
        )))
    }
}

pub(super) fn luma(rgb: &[u8]) -> u8 {
    ((19595 * u32::from(rgb[0]) + 38470 * u32::from(rgb[1]) + 7471 * u32::from(rgb[2]) + 32768)
        >> 16) as u8
}

pub(super) fn blend(base: u8, value: u8, factor: f32) -> u8 {
    // Pillow Blend.c uses float32 arithmetic, clips, then truncates.
    (f32::from(base) + factor * (f32::from(value) - f32::from(base))).clamp(0.0, 255.0) as u8
}

fn map_rgb_bytes<F>(sample: ImageSample, name: &str, map: F) -> RivetResult<ImageSample>
where
    F: FnMut(u8) -> u8,
{
    let sample = sample.into_decoded()?;
    let image = with_rgb_pixels(&sample.image, name, |pixels, width, height| {
        Ok(Tensor::from_exact_iter(
            pixels.iter().copied().map(map),
            [height, width, 3],
            sample.image.device(),
        )?)
    })?;
    Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
        image,
        label: sample.label,
    }))
}

fn map_rgb<F>(sample: ImageSample, name: &str, map: F) -> RivetResult<ImageSample>
where
    F: Fn([u8; 3]) -> [u8; 3],
{
    let sample = sample.into_decoded()?;
    let image = with_rgb_pixels(&sample.image, name, |pixels, width, height| {
        Ok(Tensor::from_exact_writer::<u8, _, _>(
            [height, width, 3],
            sample.image.device(),
            |out| {
                for pixel in pixels.chunks_exact(3) {
                    let rgb = map([pixel[0], pixel[1], pixel[2]]);
                    out.write_next(rgb[0])?;
                    out.write_next(rgb[1])?;
                    out.write_next(rgb[2])?;
                }
                Ok(())
            },
        )?)
    })?;
    Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
        image,
        label: sample.label,
    }))
}

// Pillow Convert.c uses float32 temporaries, with double expressions where
// the C constants are doubles. Preserve those boundaries for byte-exact HSV.
fn shift_hue([r, g, b]: [u8; 3], shift: u8) -> [u8; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max == min {
        return [r, g, b];
    }
    let range = f32::from(max - min);
    let s = range / f32::from(max);
    let rc = f32::from(max - r) / range;
    let gc = f32::from(max - g) / range;
    let bc = f32::from(max - b) / range;
    let h = if r == max {
        bc - gc
    } else if g == max {
        (2.0 + f64::from(rc) - f64::from(bc)) as f32
    } else {
        (4.0 + f64::from(gc) - f64::from(rc)) as f32
    };
    let h = ((f64::from(h) / 6.0 + 1.0) % 1.0) as f32;
    let h = ((f64::from(h) * 255.0) as u8).wrapping_add(shift);
    let s = (f64::from(s) * 255.0) as u8;
    let scaled = f64::from(h) * 6.0 / 255.0;
    let i = scaled.floor() as usize;
    let f = (scaled - i as f64) as f32;
    let fs = f64::from((f64::from(s) / 255.0) as f32);
    let v = f64::from(max);
    let p = (v * (1.0 - fs)).round().clamp(0.0, 255.0) as u8;
    let q = (v * (1.0 - f64::from((fs as f32) * f)))
        .round()
        .clamp(0.0, 255.0) as u8;
    let t = (v * (1.0 - fs * (1.0 - f64::from(f))))
        .round()
        .clamp(0.0, 255.0) as u8;
    match i % 6 {
        0 => [max, t, p],
        1 => [q, max, p],
        2 => [p, max, t],
        3 => [p, q, max],
        4 => [t, p, max],
        _ => [max, p, q],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrayscaleConfig {
    /// Output channels. `1` produces Luma-like output; `3` replicates the
    /// luma value over RGB channels.
    pub num_output_channels: u8,
}

impl GrayscaleConfig {
    pub const fn new(num_output_channels: u8) -> Self {
        Self {
            num_output_channels,
        }
    }

    pub const fn one_channel() -> Self {
        Self::new(1)
    }

    pub fn validate(&self) -> RivetResult<()> {
        if matches!(self.num_output_channels, 1 | 3) {
            Ok(())
        } else {
            Err(invalid_argument(
                "grayscale num_output_channels must be 1 or 3",
            ))
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let input = &sample.image;
        if input.dtype() != DType::U8 || input.rank() != 3 {
            return Err(invalid_argument(format!(
                "grayscale requires a rank-3 uint8 image, got dtype {:?} shape {:?}",
                input.dtype(),
                input.dims()
            )));
        }
        let dims = input.dims().to_vec();
        let (height, width, channels) = match axis_order {
            ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
            ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
        };
        if channels != 3 {
            return Err(invalid_shape(format!(
                "grayscale requires exactly 3 input channels, got {channels}"
            )));
        }

        let image = match axis_order {
            ImageAxisOrder::Hwc => input.clone(),
            ImageAxisOrder::Chw => input.permute(&[1, 2, 0])?,
        };
        let output_channels = usize::from(self.num_output_channels);
        let output_image = with_rgb_pixels(&image, "grayscale", |pixels, _, _| {
            Ok(Tensor::from_exact_writer::<u8, _, _>(
                [height, width, output_channels],
                &Device::Cpu,
                |write| {
                    for rgb in pixels.chunks_exact(3) {
                        let gray = luma(rgb);
                        write.write_next(gray)?;
                        if output_channels == 3 {
                            write.write_next(gray)?;
                            write.write_next(gray)?;
                        }
                    }
                    Ok(())
                },
            )?)
        })?;
        let output = crate::sample::image::DecodedSample {
            image: output_image,
            label: sample.label,
        };
        let output = if axis_order == ImageAxisOrder::Hwc {
            output
        } else {
            crate::sample::image::DecodedSample {
                image: output.image.permute(&[2, 0, 1])?.contiguous()?,
                label: output.label,
            }
        };
        Ok(ImageSample::Decoded(output))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorJitterConfig {
    pub brightness: f32,
    pub contrast: f32,
    pub hue: f32,
    pub saturation: f32,
}

impl ColorJitterConfig {
    /// Symmetric torchvision factor ranges, applied in a shuffled order.
    pub const fn new(brightness: f32, contrast: f32, hue: f32) -> Self {
        Self {
            brightness,
            contrast,
            hue,
            saturation: 0.0,
        }
    }
    pub const fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = saturation;
        self
    }
    pub fn validate(&self) -> RivetResult<()> {
        validate_factor(self.saturation, "color_jitter saturation")?;
        validate_factor(self.brightness, "color_jitter brightness")?;
        validate_factor(self.contrast, "color_jitter contrast")?;
        if !self.hue.is_finite() || !(0.0..=0.5).contains(&self.hue) {
            return Err(invalid_argument("color_jitter hue must be in [0, 0.5]"));
        }
        Ok(())
    }
    pub fn resolve(&self, rng: &mut RandomStream) -> RivetResult<ColorJitterParams> {
        let mut order = [0, 1, 2, 3];
        let count = if self.saturation == 0.0 { 3 } else { 4 };
        for i in (1..count).rev() {
            let j = rng.gen_range_usize(0..i + 1)?;
            order.swap(i, j);
        }
        let factor = |range: f32, rng: &mut RandomStream| {
            let low = (1.0 - range).max(0.0);
            low + (1.0 + range - low) * rng.next_f32()
        };
        Ok(ColorJitterParams {
            brightness: factor(self.brightness, rng),
            contrast: factor(self.contrast, rng),
            hue: -self.hue + 2.0 * self.hue * rng.next_f32(),
            saturation: if self.saturation == 0.0 {
                1.0
            } else {
                factor(self.saturation, rng)
            },
            order,
            // torchvision skips disabled ops, especially HSV conversion at hue=0.
            enabled: [
                self.brightness != 0.0,
                self.contrast != 0.0,
                self.hue != 0.0,
                self.saturation != 0.0,
            ],
        })
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        self.resolve(rng)?.apply(sample)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorJitterParams {
    pub brightness: f32,
    pub contrast: f32,
    pub hue: f32,
    pub saturation: f32,
    pub order: [usize; 4],
    pub enabled: [bool; 4],
}

impl ColorJitterParams {
    pub fn validate(&self) -> RivetResult<()> {
        BrightnessConfig::new(self.brightness).validate()?;
        ContrastConfig::new(self.contrast).validate()?;
        HueConfig::new(self.hue).validate()?;
        SaturationConfig::new(self.saturation).validate()?;
        let mut order = self.order;
        order.sort_unstable();
        if order != [0, 1, 2, 3] {
            return Err(invalid_argument("color_jitter order must be a permutation"));
        }
        Ok(())
    }
    pub fn apply(&self, mut sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        for index in self.order {
            if !self.enabled[index] {
                continue;
            }
            sample = match index {
                0 => BrightnessConfig::new(self.brightness).apply(sample)?,
                1 => ContrastConfig::new(self.contrast).apply(sample)?,
                2 => HueConfig::new(self.hue).apply(sample)?,
                _ => SaturationConfig::new(self.saturation).apply(sample)?,
            };
        }
        Ok(sample)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomGrayscaleConfig {
    pub probability: f64,
    pub num_output_channels: u8,
}

impl RandomGrayscaleConfig {
    pub const fn new(probability: f64) -> Self {
        Self {
            probability,
            num_output_channels: 3,
        }
    }

    pub const fn with_output_channels(mut self, num_output_channels: u8) -> Self {
        self.num_output_channels = num_output_channels;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(invalid_argument(
                "random_grayscale probability must be in [0.0, 1.0]",
            ));
        }
        GrayscaleConfig::new(self.num_output_channels).validate()
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        if !rng.gen_bool(self.probability)? {
            return Ok(sample);
        }
        GrayscaleConfig::new(self.num_output_channels).apply(sample, axis_order)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BrightnessConfig, ColorJitterConfig, ContrastConfig, GrayscaleConfig, HueConfig,
        RandomGrayscaleConfig,
    };
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};
    use rivet_data::random::{OpKey, RandomContext};

    fn sample() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec(vec![255u8, 0, 0, 0, 255, 0], [1, 2, 3], &Device::Cpu).unwrap(),
            label: 4,
        })
    }

    #[test]
    fn brightness_and_contrast_materialize_u8_hwc() {
        let input = sample().into_decoded().unwrap().image;
        let output = BrightnessConfig::new(0.5)
            .apply(sample())
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert_eq!(output.label, 4);
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [127, 0, 0, 0, 127, 0]);
        assert!(!output.image.same_storage(&input));
        assert!(ContrastConfig::new(f32::NAN).validate().is_err());
    }

    #[test]
    fn hue_preserves_shape_and_dtype() {
        let output = HueConfig::new(0.25)
            .apply(sample())
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
    }

    #[test]
    fn grayscale_supports_hwc_and_chw_channel_contracts() {
        let hwc = GrayscaleConfig::one_channel()
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(hwc.image.dims(), [1, 2, 1]);
        assert_eq!(hwc.image.to_vec::<u8>().unwrap(), [76, 150]);

        let chw_input =
            Tensor::from_vec(vec![255u8, 0, 0, 255, 0, 0], [3, 1, 2], &Device::Cpu).unwrap();
        let chw = GrayscaleConfig::new(3)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: chw_input,
                    label: 0,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(chw.image.dims(), [3, 1, 2]);
        assert_eq!(
            chw.image.to_vec::<u8>().unwrap(),
            [76, 150, 76, 150, 76, 150]
        );
    }

    #[test]
    fn borrowed_color_kernels_support_offset_and_strided_rgb() {
        let base = Tensor::from_vec(
            (0..2 * 5 * 7 * 3).map(|i| i as u8).collect::<Vec<_>>(),
            [2, 5, 7, 3],
            &Device::Cpu,
        )
        .unwrap()
        .get(1)
        .unwrap();
        for view in [
            base.clone(),
            base.permute(&[1, 0, 2]).unwrap(),
            base.narrow(0, 1, 3).unwrap().narrow(1, 2, 4).unwrap(),
        ] {
            let packed =
                Tensor::from_vec(view.to_vec::<u8>().unwrap(), view.dims(), &Device::Cpu).unwrap();
            for op in 0..5 {
                let apply = |image| {
                    let sample = ImageSample::Decoded(DecodedSample { image, label: 9 });
                    match op {
                        0 => BrightnessConfig::new(1.3).apply(sample),
                        1 => ContrastConfig::new(0.7).apply(sample),
                        2 => HueConfig::new(-0.17).apply(sample),
                        3 => GrayscaleConfig::new(1).apply(sample, ImageAxisOrder::Hwc),
                        _ => GrayscaleConfig::new(3).apply(sample, ImageAxisOrder::Hwc),
                    }
                    .unwrap()
                    .into_decoded()
                    .unwrap()
                };
                let actual = apply(view.clone());
                let expected = apply(packed.clone());
                assert_eq!(
                    actual.image.to_vec::<u8>().unwrap(),
                    expected.image.to_vec::<u8>().unwrap()
                );
                assert_eq!(actual.label, 9);
                assert!(!actual.image.same_storage(&view));
            }
        }
    }

    #[test]
    fn color_jitter_is_seeded_and_keeps_rgb_contract() {
        let config = ColorJitterConfig::new(0.2, 0.25, 0.1);
        let left_ctx = SampleContext::with_random(3, RandomContext::new(9));
        let right_ctx = SampleContext::with_random(3, RandomContext::new(9));
        let mut left_rng = left_ctx.stream(OpKey::from_parts("ColorJitter", 0));
        let mut right_rng = right_ctx.stream(OpKey::from_parts("ColorJitter", 0));
        let left = config
            .apply(sample(), &mut left_rng)
            .unwrap()
            .into_decoded()
            .unwrap();
        let right = config
            .apply(sample(), &mut right_rng)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(left.image.dims(), [1, 2, 3]);
        assert_eq!(left.image.dtype(), DType::U8);
        assert_eq!(
            left.image.to_vec::<u8>().unwrap(),
            right.image.to_vec::<u8>().unwrap()
        );
    }

    #[test]
    fn random_grayscale_probability_zero_and_one_are_exact() {
        let ctx = SampleContext::new(0);
        let mut rng = ctx.stream(OpKey::from_parts("RandomGrayscale", 0));
        let unchanged = RandomGrayscaleConfig::new(0.0)
            .apply(sample(), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(unchanged.image.dims(), [1, 2, 3]);

        let ctx = SampleContext::new(0);
        let mut rng = ctx.stream(OpKey::from_parts("RandomGrayscale", 0));
        let changed = RandomGrayscaleConfig::new(1.0)
            .apply(sample(), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(changed.image.dims(), [1, 2, 3]);
    }
}

pub(super) fn random_gamma(input: &Tensor, exponent: f64) -> RivetResult<Tensor> {
    use rivet_core::CpuStorageRef;

    if !matches!(input.dims(), [h, w, c] if *h > 0 && *w > 0 && *c > 0)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "RandomGamma requires nonempty uint8 or float32 HWC input",
        ));
    }
    if exponent == 1.0 {
        return Ok(input.clone());
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => {
            let lut = GammaConfig::new(exponent, 1.0).lut();
            Tensor::from_exact_iter(
                super::contiguous_values(values, layout)?
                    .iter()
                    .map(|&v| lut[v as usize]),
                input.dims(),
                input.device(),
            )
        }
        CpuStorageRef::F32(values) => Tensor::from_exact_iter(
            super::contiguous_values(values, layout)?
                .iter()
                .map(|&v| f64::from(v.clamp(0.0, 1.0)).powf(exponent) as f32),
            input.dims(),
            input.device(),
        ),
        _ => unreachable!(),
    })?)
}

pub(super) fn random_brightness_contrast(
    input: &Tensor,
    brightness: f64,
    contrast: f64,
    brightness_by_max: bool,
    safe: bool,
) -> RivetResult<Tensor> {
    use rivet_core::CpuStorageRef;
    let [h, w, channels] = *input.dims() else {
        return Err(invalid_argument(
            "RandomBrightnessContrast expects HWC input",
        ));
    };
    if h == 0 || w == 0 || channels == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "RandomBrightnessContrast requires nonempty uint8 or float32 HWC input",
        ));
    }
    if brightness == 0.0 && contrast == 0.0 {
        return Ok(input.clone());
    }
    let multiplier = 1.0 + brightness;
    let contrast = 1.0 + contrast;
    let affine = |mean: f64| {
        let (gain, offset) = if brightness_by_max {
            (contrast, brightness)
        } else {
            (contrast * multiplier, mean * (1.0 - contrast) * multiplier)
        };
        let gain = gain.min(1.0);
        (gain, offset.clamp(0.0, 1.0 - gain))
    };
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => {
            let pixels = super::contiguous_values(values, layout)?;
            let mean = if brightness_by_max || contrast == 1.0 {
                0
            } else if channels == 3 {
                let sum: u64 = pixels.chunks_exact(3).map(|p| u64::from(luma(p))).sum();
                (sum as f64 / (h * w) as f64).round() as u8
            } else {
                let sum: u64 = pixels.iter().map(|&v| u64::from(v)).sum();
                (sum as f64 / pixels.len() as f64).round() as u8
            };
            let (gain, offset) = affine(f64::from(mean) / 255.0);
            let lut: [u8; 256] = std::array::from_fn(|v| {
                if safe {
                    (v as f64 * gain + 255.0 * offset).round() as u8
                } else if brightness_by_max {
                    (v as f64 * contrast + 255.0 * brightness).round() as u8
                } else {
                    blend(0, blend(mean, v as u8, contrast as f32), multiplier as f32)
                }
            });
            Tensor::from_exact_iter(
                pixels.iter().map(|&v| lut[v as usize]),
                input.dims(),
                input.device(),
            )
        }
        CpuStorageRef::F32(values) => {
            let pixels = super::contiguous_values(values, layout)?;
            let mean = if brightness_by_max || contrast == 1.0 {
                0.0
            } else if channels == 3 {
                pixels
                    .chunks_exact(3)
                    .map(|p| {
                        0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
                    })
                    .sum::<f64>()
                    / (h * w) as f64
            } else {
                pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / pixels.len() as f64
            };
            let (gain, offset) = affine(mean);
            Tensor::from_exact_iter(
                pixels.iter().map(|&v| {
                    let v = f64::from(v);
                    let adjusted = if safe {
                        v * gain + offset
                    } else if brightness_by_max {
                        v * contrast + brightness
                    } else {
                        (mean + (v - mean) * contrast).clamp(0.0, 1.0) * multiplier
                    };
                    adjusted.clamp(0.0, 1.0) as f32
                }),
                input.dims(),
                input.device(),
            )
        }
        _ => unreachable!(),
    })?)
}

pub(super) fn hue_saturation_value(input: &Tensor, shifts: [f64; 3]) -> RivetResult<Tensor> {
    use rivet_core::CpuStorageRef;

    let [h, w, channels] = *input.dims() else {
        return Err(invalid_argument(
            "HueSaturationValue expects rank-3 HWC input",
        ));
    };
    if h == 0
        || w == 0
        || !matches!(channels, 1 | 3)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "HueSaturationValue requires nonempty uint8 or float32 HWC grayscale or RGB input",
        ));
    }
    if shifts == [0.0; 3] || (channels == 1 && shifts[2] == 0.0) {
        return Ok(input.clone());
    }
    let [hue, saturation, value] = shifts;
    let shift_rgb = |[r, g, b]: [f64; 3]| {
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let chroma = max - min;
        let h = if chroma == 0.0 {
            0.0
        } else if max == r {
            (g - b) / chroma
        } else if max == g {
            2.0 + (b - r) / chroma
        } else {
            4.0 + (r - g) / chroma
        };
        let s = if max == 0.0 { 0.0 } else { chroma / max };
        let h = (h + hue / 60.0).rem_euclid(6.0);
        let v = (max + value / 255.0).clamp(0.0, 1.0);
        let c = v * (s + saturation / 255.0).clamp(0.0, 1.0);
        let x = c * (1.0 - (h % 2.0 - 1.0).abs());
        let rgb = match h as u32 {
            0 => [c, x, 0.0],
            1 => [x, c, 0.0],
            2 => [0.0, c, x],
            3 => [0.0, x, c],
            4 => [x, 0.0, c],
            _ => [c, 0.0, x],
        };
        rgb.map(|v0| v0 + (v - c))
    };
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_hsv {
            ($values:expr, $normalizer:expr, $finish:expr) => {{
                let pixels = super::contiguous_values($values, layout)?;
                let finish = $finish;
                if channels == 1 {
                    Tensor::from_exact_iter(
                        pixels.iter().map(|&v| {
                            finish((f64::from(v) * $normalizer + value / 255.0).clamp(0.0, 1.0))
                        }),
                        input.dims(),
                        input.device(),
                    )
                } else {
                    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                        for p in pixels.chunks_exact(3) {
                            let rgb = [p[0], p[1], p[2]]
                                .map(|v| (f64::from(v) * $normalizer).clamp(0.0, 1.0));
                            for v in shift_rgb(rgb) {
                                out.write_next(finish(v))?;
                            }
                        }
                        Ok(())
                    })
                }
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_hsv!(values, 1.0 / 255.0, |v: f64| (v * 255.0).round() as u8)
            }
            CpuStorageRef::F32(values) => write_hsv!(values, 1.0, |v| v as f32),
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn to_sepia(input: &Tensor, alpha: f64) -> RivetResult<Tensor> {
    let [h, w, channels] = *input.dims() else {
        return Err(invalid_argument("ToSepia expects rank-3 HWC input"));
    };
    if h == 0
        || w == 0
        || !matches!(channels, 1 | 3)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "ToSepia requires nonempty uint8 or float32 HWC grayscale or RGB input",
        ));
    }
    if channels == 1 || alpha == 0.0 {
        return Ok(input.clone());
    }
    let matrix = [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ];
    super::weather::write_weather_rgb(input, |_, _, rgb| {
        std::array::from_fn(|c| {
            let [r, g, b] = matrix[c];
            let sepia = r * rgb[0] + g * rgb[1] + b * rgb[2];
            rgb[c] + alpha * (sepia - rgb[c])
        })
    })
}

type AnchorRange = ([u8; 3], [u8; 3]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorizeConfig {
    pub black_range: AnchorRange,
    pub white_range: AnchorRange,
    pub mid_range: Option<AnchorRange>,
    pub mid_value_range: (u8, u8),
}

impl ColorizeConfig {
    pub(super) fn valid(&self) -> bool {
        let ordered = |(lo, hi): AnchorRange| lo.into_iter().zip(hi).all(|(lo, hi)| lo <= hi);
        ordered(self.black_range)
            && ordered(self.white_range)
            && self.mid_range.is_none_or(|range| {
                let (lo, hi) = self.mid_value_range;
                ordered(range) && lo > 0 && lo <= hi && hi < 255
            })
    }

    pub(super) fn apply(&self, input: &Tensor, rng: &mut RandomStream) -> RivetResult<Tensor> {
        use rivet_core::CpuStorageRef;

        let [h, w, 1] = *input.dims() else {
            return Err(invalid_argument("Colorize requires HWC grayscale input"));
        };
        if h == 0 || w == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "Colorize requires nonempty uint8 or float32 HWC grayscale input",
            ));
        }
        let sample_rgb = |(lo, hi): AnchorRange, rng: &mut RandomStream| -> RivetResult<[f64; 3]> {
            let mut rgb = [0.0; 3];
            for (i, value) in rgb.iter_mut().enumerate() {
                *value = rng.gen_range_usize(lo[i] as usize..hi[i] as usize + 1)? as f64;
            }
            Ok(rgb)
        };
        let black = sample_rgb(self.black_range, rng)?;
        let white = sample_rgb(self.white_range, rng)?;
        let mid = if let Some(range) = self.mid_range {
            let (lo, hi) = self.mid_value_range;
            Some((
                rng.gen_range_usize(lo as usize..hi as usize + 1)? as f64,
                sample_rgb(range, rng)?,
            ))
        } else {
            None
        };
        let map = |value: f64| {
            let (a, b, t) = match mid {
                Some((pivot, rgb)) if value <= pivot => (black, rgb, value / pivot),
                Some((pivot, rgb)) => (rgb, white, (value - pivot) / (255.0 - pivot)),
                None => (black, white, value / 255.0),
            };
            std::array::from_fn::<_, 3, _>(|i| a[i] + t * (b[i] - a[i]))
        };
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(values) => {
                let lut: [[u8; 3]; 256] =
                    std::array::from_fn(|v| map(v as f64).map(|c| c.round() as u8));
                let values = super::contiguous_values(values, layout)?;
                Tensor::from_exact_writer([h, w, 3], input.device(), |out| {
                    for &value in values {
                        for v in lut[value as usize] {
                            out.write_next(v)?;
                        }
                    }
                    Ok(())
                })
            }
            CpuStorageRef::F32(values) => {
                let values = super::contiguous_values(values, layout)?;
                Tensor::from_exact_writer([h, w, 3], input.device(), |out| {
                    for &value in values {
                        for v in map(f64::from(value.clamp(0.0, 1.0)) * 255.0) {
                            out.write_next((v / 255.0) as f32)?;
                        }
                    }
                    Ok(())
                })
            }
            _ => unreachable!(),
        })?)
    }
}

pub(super) fn exposure_matching(
    input: &Tensor,
    target: f64,
    gain_range: Option<(f64, f64)>,
) -> RivetResult<Tensor> {
    use rivet_core::CpuStorageRef;
    if !matches!(input.dims(), [h, w, c] if *h > 0 && *w > 0 && *c > 0)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "ExposureMatching requires nonempty uint8 or float32 HWC input",
        ));
    }
    let gain = |mean: f64| {
        let ratio = target / mean;
        gain_range.map_or(ratio, |(lo, hi)| ratio.clamp(lo, hi))
    };
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => {
            let pixels = super::contiguous_values(values, layout)?;
            let sum: u64 = pixels.iter().map(|&v| u64::from(v)).sum();
            if sum == 0 {
                return Ok(input.clone());
            }
            let gain = gain((sum as f64 / pixels.len() as f64) / 255.0);
            if gain == 1.0 {
                return Ok(input.clone());
            }
            let lut: [u8; 256] = std::array::from_fn(|v| (v as f64 * gain).round() as u8);
            Tensor::from_exact_iter(
                pixels.iter().map(|&v| lut[v as usize]),
                input.dims(),
                input.device(),
            )
        }
        CpuStorageRef::F32(values) => {
            let pixels = super::contiguous_values(values, layout)?;
            let sum = pixels.iter().map(|&v| f64::from(v)).sum::<f64>();
            if sum <= 0.0 {
                return Ok(input.clone());
            }
            let gain = gain(sum / pixels.len() as f64);
            if gain == 1.0 {
                return Ok(input.clone());
            }
            Tensor::from_exact_iter(
                pixels
                    .iter()
                    .map(|&v| (f64::from(v) * gain).clamp(0.0, 1.0) as f32),
                input.dims(),
                input.device(),
            )
        }
        _ => unreachable!(),
    })?)
}
