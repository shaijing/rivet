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
}

impl ColorJitterConfig {
    /// Symmetric torchvision factor ranges, applied in a shuffled order.
    pub const fn new(brightness: f32, contrast: f32, hue: f32) -> Self {
        Self {
            brightness,
            contrast,
            hue,
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        validate_factor(self.brightness, "color_jitter brightness")?;
        validate_factor(self.contrast, "color_jitter contrast")?;
        if !self.hue.is_finite() || !(0.0..=0.5).contains(&self.hue) {
            return Err(invalid_argument("color_jitter hue must be in [0, 0.5]"));
        }
        Ok(())
    }
    pub fn resolve(&self, rng: &mut RandomStream) -> RivetResult<ColorJitterParams> {
        let mut order = [0, 1, 2];
        for i in (1..3).rev() {
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
            order,
            // torchvision skips disabled ops, especially HSV conversion at hue=0.
            enabled: [
                self.brightness != 0.0,
                self.contrast != 0.0,
                self.hue != 0.0,
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
    pub order: [usize; 3],
    pub enabled: [bool; 3],
}

impl ColorJitterParams {
    pub fn validate(&self) -> RivetResult<()> {
        BrightnessConfig::new(self.brightness).validate()?;
        ContrastConfig::new(self.contrast).validate()?;
        HueConfig::new(self.hue).validate()?;
        let mut order = self.order;
        order.sort_unstable();
        if order != [0, 1, 2] {
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
                _ => HueConfig::new(self.hue).apply(sample)?,
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
