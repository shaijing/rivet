//! Native torchvision augmentation policies. Sampling uses Rivet streams;
//! resolved operations can be executed independently for differential tests.
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::{
    AutocontrastConfig, BrightnessConfig, ContrastConfig, EqualizeConfig, InterpolationMode,
    InvertConfig, PosterizeConfig, SaturationConfig, SharpnessConfig, SolarizeConfig,
    with_rgb_pixels,
};
use rivet_core::{CpuStorageRef, Device, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AugmentOp {
    Identity,
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
    Rotate,
    Brightness,
    Color,
    Contrast,
    Sharpness,
    Posterize,
    Solarize,
    AutoContrast,
    Equalize,
    Invert,
}
impl AugmentOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::ShearX => "ShearX",
            Self::ShearY => "ShearY",
            Self::TranslateX => "TranslateX",
            Self::TranslateY => "TranslateY",
            Self::Rotate => "Rotate",
            Self::Brightness => "Brightness",
            Self::Color => "Color",
            Self::Contrast => "Contrast",
            Self::Sharpness => "Sharpness",
            Self::Posterize => "Posterize",
            Self::Solarize => "Solarize",
            Self::AutoContrast => "AutoContrast",
            Self::Equalize => "Equalize",
            Self::Invert => "Invert",
        }
    }
    pub fn parse(name: &str) -> RivetResult<Self> {
        ALL.into_iter()
            .chain([Self::Invert])
            .find(|op| op.name() == name)
            .ok_or_else(|| invalid_argument(format!("unknown augmentation operation {name}")))
    }
}
const ALL: [AugmentOp; 14] = [
    AugmentOp::Identity,
    AugmentOp::ShearX,
    AugmentOp::ShearY,
    AugmentOp::TranslateX,
    AugmentOp::TranslateY,
    AugmentOp::Rotate,
    AugmentOp::Brightness,
    AugmentOp::Color,
    AugmentOp::Contrast,
    AugmentOp::Sharpness,
    AugmentOp::Posterize,
    AugmentOp::Solarize,
    AugmentOp::AutoContrast,
    AugmentOp::Equalize,
];
const MIX: [AugmentOp; 13] = [
    AugmentOp::ShearX,
    AugmentOp::ShearY,
    AugmentOp::TranslateX,
    AugmentOp::TranslateY,
    AugmentOp::Rotate,
    AugmentOp::Posterize,
    AugmentOp::Solarize,
    AugmentOp::AutoContrast,
    AugmentOp::Equalize,
    AugmentOp::Brightness,
    AugmentOp::Color,
    AugmentOp::Contrast,
    AugmentOp::Sharpness,
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AugmentStep {
    pub op: AugmentOp,
    pub magnitude: f32,
}
impl AugmentStep {
    pub fn apply(
        &self,
        sample: ImageSample,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> RivetResult<ImageSample> {
        self.apply_backend::<false>(sample, interpolation, fill)
    }
    pub(crate) fn apply_backend<const TENSOR: bool>(
        &self,
        sample: ImageSample,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> RivetResult<ImageSample> {
        use AugmentOp::*;
        let m = self.magnitude;
        if !m.is_finite() {
            return Err(invalid_argument("augmentation magnitude must be finite"));
        }
        if self.op == AugmentOp::Posterize && !(0.0..=8.0).contains(&m) {
            return Err(invalid_argument("posterize magnitude must be in [0,8]"));
        }
        if self.op == AugmentOp::Solarize && !(0.0..=255.0).contains(&m) {
            return Err(invalid_argument("solarize magnitude must be in [0,255]"));
        }
        match self.op {
            Identity => Ok(sample),
            ShearX | ShearY | TranslateX | TranslateY | Rotate => {
                let kind = match self.op {
                    ShearX => 0,
                    ShearY => 1,
                    TranslateX => 2,
                    TranslateY => 3,
                    _ => 4,
                };
                crate::transforms::spatial::advanced_geometry::policy_warp::<TENSOR>(
                    sample,
                    kind,
                    f64::from(m),
                    interpolation,
                    fill,
                )
            }
            Brightness => BrightnessConfig::new(1.0 + m).apply(sample),
            Color | Contrast | Sharpness | AutoContrast if TENSOR => {
                tensor_color(sample, self.op, m)
            }
            Color => SaturationConfig::new(1.0 + m).apply(sample),
            Contrast => ContrastConfig::new(1.0 + m).apply(sample),
            Sharpness => SharpnessConfig::new(1.0 + m).apply(sample, ImageAxisOrder::Hwc),
            Posterize => PosterizeConfig::new(m as u8).apply(sample, ImageAxisOrder::Hwc),
            Solarize => SolarizeConfig::new(m.ceil().clamp(0.0, 255.0) as u8)
                .apply(sample, ImageAxisOrder::Hwc),
            AutoContrast => AutocontrastConfig::new().apply(sample, ImageAxisOrder::Hwc),
            Equalize => EqualizeConfig::new().apply(sample, ImageAxisOrder::Hwc),
            Invert => InvertConfig::new().apply(sample, ImageAxisOrder::Hwc),
        }
    }
}
fn tensor_color(sample: ImageSample, op: AugmentOp, m: f32) -> RivetResult<ImageSample> {
    let sample = sample.into_decoded()?;
    let image = with_rgb_pixels(&sample.image, "AugMix color", |pixels, w, h| {
        let gray = |p: &[u8]| {
            (0.2989f32 * f32::from(p[0]) + 0.587f32 * f32::from(p[1]) + 0.114f32 * f32::from(p[2]))
                as u8
        };
        let mean = if op == AugmentOp::Contrast {
            pixels
                .chunks_exact(3)
                .map(|p| u64::from(gray(p)))
                .sum::<u64>() as f32
                / (w * h) as f32
        } else {
            0.0
        };
        let mut min = [255u8; 3];
        let mut max = [0u8; 3];
        if op == AugmentOp::AutoContrast {
            for rgb in pixels.chunks_exact(3) {
                for c in 0..3 {
                    min[c] = min[c].min(rgb[c]);
                    max[c] = max[c].max(rgb[c]);
                }
            }
        }
        let ratio = (1.0 + f64::from(m)) as f32;
        let inverse = (-f64::from(m)) as f32;
        Ok(Tensor::from_exact_writer::<u8, _, _>(
            [h, w, 3],
            &Device::Cpu,
            |out| {
                for y in 0..h {
                    for x in 0..w {
                        let offset = (y * w + x) * 3;
                        let rgb = &pixels[offset..offset + 3];
                        let luma = if op == AugmentOp::Color {
                            f32::from(gray(rgb))
                        } else {
                            mean
                        };
                        for c in 0..3 {
                            let value = f32::from(rgb[c]);
                            let result = match op {
                                AugmentOp::AutoContrast => {
                                    if min[c] == max[c] {
                                        value
                                    } else {
                                        (value - f32::from(min[c]))
                                            * (255.0 / f32::from(max[c] - min[c]))
                                    }
                                }
                                AugmentOp::Sharpness => {
                                    let blurred = if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                                        rgb[c]
                                    } else {
                                        let mut sum = 0.0f32;
                                        for sy in y - 1..=y + 1 {
                                            for sx in x - 1..=x + 1 {
                                                let weight = if sy == y && sx == x {
                                                    5.0f32 / 13.0
                                                } else {
                                                    1.0f32 / 13.0
                                                };
                                                sum += f32::from(pixels[(sy * w + sx) * 3 + c])
                                                    * weight;
                                            }
                                        }
                                        sum.round_ties_even().clamp(0.0, 255.0) as u8
                                    };
                                    ratio * value + inverse * f32::from(blurred)
                                }
                                _ => ratio * value + inverse * luma,
                            };
                            out.write_next(result.clamp(0.0, 255.0) as u8)?;
                        }
                    }
                }
                Ok(())
            },
        )?)
    })?;
    Ok(ImageSample::Decoded(DecodedSample {
        image,
        label: sample.label,
    }))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoAugmentPolicy {
    ImageNet,
    Cifar10,
    Svhn,
}
impl std::str::FromStr for AutoAugmentPolicy {
    type Err = crate::errors::VisionError;
    fn from_str(s: &str) -> RivetResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "imagenet" => Ok(Self::ImageNet),
            "cifar10" => Ok(Self::Cifar10),
            "svhn" => Ok(Self::Svhn),
            _ => Err(invalid_argument("policy must be imagenet, cifar10 or svhn")),
        }
    }
}
fn check_interpolation(mode: InterpolationMode) -> RivetResult<()> {
    if matches!(
        mode,
        InterpolationMode::Nearest | InterpolationMode::Bilinear | InterpolationMode::Bicubic
    ) {
        Ok(())
    } else {
        Err(invalid_argument(
            "augmentation geometry requires nearest, bilinear or bicubic interpolation",
        ))
    }
}
// torch.linspace float32 uses endpoint-based evaluation to preserve accuracy.
fn linspace(start: f32, end: f32, bins: usize, index: usize) -> f32 {
    let step = (end - start) / (bins - 1) as f32;
    if index < bins / 2 {
        start + step * index as f32
    } else {
        end - step * (bins - index - 1) as f32
    }
}
fn magnitude(
    op: AugmentOp,
    bins: usize,
    index: usize,
    width: usize,
    height: usize,
    wide: bool,
    mix: bool,
) -> (f32, bool) {
    use AugmentOp::*;
    let upper = |normal, wide_value| if wide { wide_value } else { normal };
    match op {
        ShearX | ShearY => (linspace(0.0, upper(0.3, 0.99), bins, index), true),
        TranslateX | TranslateY => {
            let size = if op == TranslateX { width } else { height } as f64;
            let max = if wide {
                32.0
            } else if mix {
                size / 3.0
            } else {
                150.0 / 331.0 * size
            };
            (linspace(0.0, max as f32, bins, index), true)
        }
        Rotate => (linspace(0.0, upper(30.0, 135.0), bins, index), true),
        Brightness | Color | Contrast | Sharpness => {
            (linspace(0.0, upper(0.9, 0.99), bins, index), true)
        }
        Posterize => {
            let max = if mix { 4.0 } else { 8.0 };
            let span = if wide { 6.0 } else { 4.0 };
            (
                max - (index as f32 / ((bins - 1) as f32 / span)).round_ties_even(),
                false,
            )
        }
        Solarize => (linspace(255.0, 0.0, bins, index), false),
        _ => (0.0, false),
    }
}
fn run_steps(
    mut sample: ImageSample,
    steps: &[AugmentStep],
    interpolation: InterpolationMode,
    fill: u8,
) -> RivetResult<ImageSample> {
    for step in steps {
        sample = step.apply(sample, interpolation, fill)?;
    }
    Ok(sample)
}
fn dimensions(sample: &ImageSample) -> RivetResult<(usize, usize)> {
    let ImageSample::Decoded(decoded) = sample else {
        return Err(invalid_argument(
            "augmentation policies require a decoded image",
        ));
    };
    let d = &decoded.image;
    if d.dims().len() != 3
        || d.dtype() != rivet_core::DType::U8
        || d.dims()[2] != 3
        || d.dims()[0] == 0
        || d.dims()[1] == 0
    {
        return Err(invalid_argument(
            "augmentation policies require a nonempty uint8 HWC RGB image",
        ));
    }
    Ok((d.dims()[1], d.dims()[0]))
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoAugmentConfig {
    pub policy: AutoAugmentPolicy,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}
impl AutoAugmentConfig {
    pub const fn new(policy: AutoAugmentPolicy) -> Self {
        Self {
            policy,
            interpolation: InterpolationMode::Nearest,
            fill: 0,
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        check_interpolation(self.interpolation)
    }
    pub fn resolve(
        &self,
        width: usize,
        height: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Vec<AugmentStep>> {
        self.validate()?;
        let table = match self.policy {
            AutoAugmentPolicy::ImageNet => &IMAGENET,
            AutoAugmentPolicy::Cifar10 => &CIFAR10,
            AutoAugmentPolicy::Svhn => &SVHN,
        };
        let pair = &table[rng.gen_range_usize(0..table.len())?];
        let probs = [rng.next_f32(), rng.next_f32()];
        let signs = [rng.gen_range_usize(0..2)?, rng.gen_range_usize(0..2)?];
        let mut steps = Vec::new();
        for (i, &(op, p, index)) in pair.iter().enumerate() {
            if probs[i] <= p {
                let (mut m, signed) = magnitude(op, 10, index, width, height, false, false);
                if signed && signs[i] == 0 {
                    m = -m;
                }
                steps.push(AugmentStep { op, magnitude: m });
            }
        }
        Ok(steps)
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        let (w, h) = dimensions(&sample)?;
        run_steps(
            sample,
            &self.resolve(w, h, rng)?,
            self.interpolation,
            self.fill,
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandAugmentConfig {
    pub num_ops: usize,
    pub magnitude: usize,
    pub num_magnitude_bins: usize,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}
impl Default for RandAugmentConfig {
    fn default() -> Self {
        Self {
            num_ops: 2,
            magnitude: 9,
            num_magnitude_bins: 31,
            interpolation: InterpolationMode::Nearest,
            fill: 0,
        }
    }
}
impl RandAugmentConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if self.num_magnitude_bins < 2 || self.magnitude >= self.num_magnitude_bins {
            return Err(invalid_argument(
                "RandAugment magnitude must be below num_magnitude_bins >= 2",
            ));
        }
        check_interpolation(self.interpolation)
    }
    pub fn resolve(
        &self,
        width: usize,
        height: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Vec<AugmentStep>> {
        self.validate()?;
        (0..self.num_ops)
            .map(|_| {
                let op = ALL[rng.gen_range_usize(0..ALL.len())?];
                let (mut m, signed) = magnitude(
                    op,
                    self.num_magnitude_bins,
                    self.magnitude,
                    width,
                    height,
                    false,
                    false,
                );
                if signed && rng.gen_bool(0.5)? {
                    m = -m;
                }
                Ok(AugmentStep { op, magnitude: m })
            })
            .collect()
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        let (w, h) = dimensions(&sample)?;
        run_steps(
            sample,
            &self.resolve(w, h, rng)?,
            self.interpolation,
            self.fill,
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrivialAugmentWideConfig {
    pub num_magnitude_bins: usize,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}
impl Default for TrivialAugmentWideConfig {
    fn default() -> Self {
        Self {
            num_magnitude_bins: 31,
            interpolation: InterpolationMode::Nearest,
            fill: 0,
        }
    }
}
impl TrivialAugmentWideConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if self.num_magnitude_bins < 2 {
            return Err(invalid_argument(
                "TrivialAugmentWide requires num_magnitude_bins >= 2",
            ));
        }
        check_interpolation(self.interpolation)
    }
    pub fn resolve(
        &self,
        width: usize,
        height: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Vec<AugmentStep>> {
        self.validate()?;
        let op = ALL[rng.gen_range_usize(0..ALL.len())?];
        let index = if matches!(
            op,
            AugmentOp::Identity | AugmentOp::AutoContrast | AugmentOp::Equalize
        ) {
            0
        } else {
            rng.gen_range_usize(0..self.num_magnitude_bins)?
        };
        let (mut m, signed) = magnitude(
            op,
            self.num_magnitude_bins,
            index,
            width,
            height,
            true,
            false,
        );
        if signed && rng.gen_bool(0.5)? {
            m = -m;
        }
        Ok(vec![AugmentStep { op, magnitude: m }])
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        let (w, h) = dimensions(&sample)?;
        run_steps(
            sample,
            &self.resolve(w, h, rng)?,
            self.interpolation,
            self.fill,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AugMixConfig {
    pub severity: usize,
    pub mixture_width: usize,
    pub chain_depth: i32,
    pub alpha: f32,
    pub all_ops: bool,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}
impl Default for AugMixConfig {
    fn default() -> Self {
        Self {
            severity: 3,
            mixture_width: 3,
            chain_depth: -1,
            alpha: 1.0,
            all_ops: true,
            interpolation: InterpolationMode::Bilinear,
            fill: 0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct AugMixPlan {
    pub original_weight: f32,
    pub chain_weights: Vec<f32>,
    pub chains: Vec<Vec<AugmentStep>>,
}
impl AugMixConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if !(1..=10).contains(&self.severity)
            || self.mixture_width == 0
            || !self.alpha.is_finite()
            || self.alpha <= 0.0
        {
            return Err(invalid_argument(
                "AugMix requires severity 1..=10, positive mixture_width and finite positive alpha",
            ));
        }
        if !matches!(
            self.interpolation,
            InterpolationMode::Nearest | InterpolationMode::Bilinear
        ) {
            return Err(invalid_argument(
                "AugMix Tensor sampling supports nearest or bilinear interpolation",
            ));
        }
        Ok(())
    }
    pub fn resolve(&self, w: usize, h: usize, rng: &mut RandomStream) -> RivetResult<AugMixPlan> {
        self.validate()?;
        let beta = dirichlet(2, self.alpha, rng);
        let weights = dirichlet(self.mixture_width, self.alpha, rng);
        let chain_weights = weights.into_iter().map(|weight| weight * beta[1]).collect();
        let ops = &MIX[..if self.all_ops { 13 } else { 9 }];
        let mut chains = Vec::with_capacity(self.mixture_width);
        for _ in 0..self.mixture_width {
            let depth = if self.chain_depth > 0 {
                self.chain_depth as usize
            } else {
                rng.gen_range_usize(1..4)?
            };
            let mut steps = Vec::with_capacity(depth);
            for _ in 0..depth {
                let op = ops[rng.gen_range_usize(0..ops.len())?];
                let index = if matches!(op, AugmentOp::AutoContrast | AugmentOp::Equalize) {
                    0
                } else {
                    rng.gen_range_usize(0..self.severity)?
                };
                let (mut m, signed) = magnitude(op, 10, index, w, h, false, true);
                if signed && rng.gen_bool(0.5)? {
                    m = -m;
                }
                steps.push(AugmentStep { op, magnitude: m });
            }
            chains.push(steps);
        }
        Ok(AugMixPlan {
            original_weight: beta[0],
            chain_weights,
            chains,
        })
    }
    pub fn apply_plan(&self, sample: ImageSample, plan: &AugMixPlan) -> RivetResult<ImageSample> {
        self.validate()?;
        let (w, h) = dimensions(&sample)?;
        let total = plan.original_weight + plan.chain_weights.iter().sum::<f32>();
        if (total - 1.0).abs() > 1e-4
            || plan.chains.len() != plan.chain_weights.len()
            || !plan.original_weight.is_finite()
            || plan.original_weight < 0.0
            || plan
                .chain_weights
                .iter()
                .any(|w| !w.is_finite() || *w < 0.0)
        {
            return Err(invalid_argument("invalid AugMix mixing plan"));
        }
        let original = sample.clone().into_decoded()?;
        let mut mix = with_rgb_pixels(&original.image, "AugMix", |pixels, _, _| {
            Ok(Tensor::from_exact_iter(
                pixels.iter().map(|&v| f32::from(v) * plan.original_weight),
                [h, w, 3],
                &Device::Cpu,
            )?)
        })?;
        for (chain, &weight) in plan.chains.iter().zip(&plan.chain_weights) {
            let mut aug = sample.clone();
            for step in chain {
                aug = step.apply_backend::<true>(aug, self.interpolation, self.fill)?;
            }
            let aug = aug.into_decoded()?;
            mix = with_rgb_pixels(&aug.image, "AugMix", |pixels, _, _| {
                Ok(mix.with_cpu_storage(|storage, _| {
                    let CpuStorageRef::F32(values) = storage else {
                        unreachable!()
                    };
                    Tensor::from_exact_iter(
                        values
                            .iter()
                            .zip(pixels)
                            .map(|(&value, &pixel)| value + weight * f32::from(pixel)),
                        [h, w, 3],
                        &Device::Cpu,
                    )
                })?)
            })?;
        }
        let image = mix.with_cpu_storage(|storage, _| {
            let CpuStorageRef::F32(values) = storage else {
                unreachable!()
            };
            Tensor::from_exact_iter(
                values.iter().map(|&v| v.clamp(0.0, 255.0) as u8),
                [h, w, 3],
                &Device::Cpu,
            )
        })?;
        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: original.label,
        }))
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        let (w, h) = dimensions(&sample)?;
        self.apply_plan(sample, &self.resolve(w, h, rng)?)
    }
}
fn dirichlet(n: usize, alpha: f32, rng: &mut RandomStream) -> Vec<f32> {
    let values: Vec<f64> = (0..n)
        .map(|_| gamma_sample(f64::from(alpha), rng).max(f64::MIN_POSITIVE))
        .collect();
    let sum: f64 = values.iter().sum();
    values.into_iter().map(|v| (v / sum) as f32).collect()
}
fn gamma_sample(alpha: f64, rng: &mut RandomStream) -> f64 {
    if alpha < 1.0 {
        return gamma_sample(alpha + 1.0, rng)
            * rng.next_f64().max(f64::MIN_POSITIVE).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        let x = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * rng.next_f64()).cos();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// torchvision.transforms.autoaugment.AutoAugment._get_policies.
const IMAGENET: [[(AugmentOp, f32, usize); 2]; 25] = [
    [(AugmentOp::Posterize, 0.4, 8), (AugmentOp::Rotate, 0.6, 9)],
    [
        (AugmentOp::Solarize, 0.6, 5),
        (AugmentOp::AutoContrast, 0.6, 0),
    ],
    [(AugmentOp::Equalize, 0.8, 0), (AugmentOp::Equalize, 0.6, 0)],
    [
        (AugmentOp::Posterize, 0.6, 7),
        (AugmentOp::Posterize, 0.6, 6),
    ],
    [(AugmentOp::Equalize, 0.4, 0), (AugmentOp::Solarize, 0.2, 4)],
    [(AugmentOp::Equalize, 0.4, 0), (AugmentOp::Rotate, 0.8, 8)],
    [(AugmentOp::Solarize, 0.6, 3), (AugmentOp::Equalize, 0.6, 0)],
    [
        (AugmentOp::Posterize, 0.8, 5),
        (AugmentOp::Equalize, 1.0, 0),
    ],
    [(AugmentOp::Rotate, 0.2, 3), (AugmentOp::Solarize, 0.6, 8)],
    [
        (AugmentOp::Equalize, 0.6, 0),
        (AugmentOp::Posterize, 0.4, 6),
    ],
    [(AugmentOp::Rotate, 0.8, 8), (AugmentOp::Color, 0.4, 0)],
    [(AugmentOp::Rotate, 0.4, 9), (AugmentOp::Equalize, 0.6, 0)],
    [(AugmentOp::Equalize, 0.0, 0), (AugmentOp::Equalize, 0.8, 0)],
    [(AugmentOp::Invert, 0.6, 0), (AugmentOp::Equalize, 1.0, 0)],
    [(AugmentOp::Color, 0.6, 4), (AugmentOp::Contrast, 1.0, 8)],
    [(AugmentOp::Rotate, 0.8, 8), (AugmentOp::Color, 1.0, 2)],
    [(AugmentOp::Color, 0.8, 8), (AugmentOp::Solarize, 0.8, 7)],
    [(AugmentOp::Sharpness, 0.4, 7), (AugmentOp::Invert, 0.6, 0)],
    [(AugmentOp::ShearX, 0.6, 5), (AugmentOp::Equalize, 1.0, 0)],
    [(AugmentOp::Color, 0.4, 0), (AugmentOp::Equalize, 0.6, 0)],
    [(AugmentOp::Equalize, 0.4, 0), (AugmentOp::Solarize, 0.2, 4)],
    [
        (AugmentOp::Solarize, 0.6, 5),
        (AugmentOp::AutoContrast, 0.6, 0),
    ],
    [(AugmentOp::Invert, 0.6, 0), (AugmentOp::Equalize, 1.0, 0)],
    [(AugmentOp::Color, 0.6, 4), (AugmentOp::Contrast, 1.0, 8)],
    [(AugmentOp::Equalize, 0.8, 0), (AugmentOp::Equalize, 0.6, 0)],
];
const CIFAR10: [[(AugmentOp, f32, usize); 2]; 25] = [
    [(AugmentOp::Invert, 0.1, 0), (AugmentOp::Contrast, 0.2, 6)],
    [(AugmentOp::Rotate, 0.7, 2), (AugmentOp::TranslateX, 0.3, 9)],
    [
        (AugmentOp::Sharpness, 0.8, 1),
        (AugmentOp::Sharpness, 0.9, 3),
    ],
    [(AugmentOp::ShearY, 0.5, 8), (AugmentOp::TranslateY, 0.7, 9)],
    [
        (AugmentOp::AutoContrast, 0.5, 0),
        (AugmentOp::Equalize, 0.9, 0),
    ],
    [(AugmentOp::ShearY, 0.2, 7), (AugmentOp::Posterize, 0.3, 7)],
    [(AugmentOp::Color, 0.4, 3), (AugmentOp::Brightness, 0.6, 7)],
    [
        (AugmentOp::Sharpness, 0.3, 9),
        (AugmentOp::Brightness, 0.7, 9),
    ],
    [(AugmentOp::Equalize, 0.6, 0), (AugmentOp::Equalize, 0.5, 0)],
    [
        (AugmentOp::Contrast, 0.6, 7),
        (AugmentOp::Sharpness, 0.6, 5),
    ],
    [(AugmentOp::Color, 0.7, 7), (AugmentOp::TranslateX, 0.5, 8)],
    [
        (AugmentOp::Equalize, 0.3, 0),
        (AugmentOp::AutoContrast, 0.4, 0),
    ],
    [
        (AugmentOp::TranslateY, 0.4, 3),
        (AugmentOp::Sharpness, 0.2, 6),
    ],
    [(AugmentOp::Brightness, 0.9, 6), (AugmentOp::Color, 0.2, 8)],
    [(AugmentOp::Solarize, 0.5, 2), (AugmentOp::Invert, 0.0, 0)],
    [
        (AugmentOp::Equalize, 0.2, 0),
        (AugmentOp::AutoContrast, 0.6, 0),
    ],
    [(AugmentOp::Equalize, 0.2, 0), (AugmentOp::Equalize, 0.6, 0)],
    [(AugmentOp::Color, 0.9, 9), (AugmentOp::Equalize, 0.6, 0)],
    [
        (AugmentOp::AutoContrast, 0.8, 0),
        (AugmentOp::Solarize, 0.2, 8),
    ],
    [(AugmentOp::Brightness, 0.1, 3), (AugmentOp::Color, 0.7, 0)],
    [
        (AugmentOp::Solarize, 0.4, 5),
        (AugmentOp::AutoContrast, 0.9, 0),
    ],
    [
        (AugmentOp::TranslateY, 0.9, 9),
        (AugmentOp::TranslateY, 0.7, 9),
    ],
    [
        (AugmentOp::AutoContrast, 0.9, 0),
        (AugmentOp::Solarize, 0.8, 3),
    ],
    [(AugmentOp::Equalize, 0.8, 0), (AugmentOp::Invert, 0.1, 0)],
    [
        (AugmentOp::TranslateY, 0.7, 9),
        (AugmentOp::AutoContrast, 0.9, 0),
    ],
];
const SVHN: [[(AugmentOp, f32, usize); 2]; 25] = [
    [(AugmentOp::ShearX, 0.9, 4), (AugmentOp::Invert, 0.2, 0)],
    [(AugmentOp::ShearY, 0.9, 8), (AugmentOp::Invert, 0.7, 0)],
    [(AugmentOp::Equalize, 0.6, 0), (AugmentOp::Solarize, 0.6, 6)],
    [(AugmentOp::Invert, 0.9, 0), (AugmentOp::Equalize, 0.6, 0)],
    [(AugmentOp::Equalize, 0.6, 0), (AugmentOp::Rotate, 0.9, 3)],
    [
        (AugmentOp::ShearX, 0.9, 4),
        (AugmentOp::AutoContrast, 0.8, 0),
    ],
    [(AugmentOp::ShearY, 0.9, 8), (AugmentOp::Invert, 0.4, 0)],
    [(AugmentOp::ShearY, 0.9, 5), (AugmentOp::Solarize, 0.2, 6)],
    [
        (AugmentOp::Invert, 0.9, 0),
        (AugmentOp::AutoContrast, 0.8, 0),
    ],
    [(AugmentOp::Equalize, 0.6, 0), (AugmentOp::Rotate, 0.9, 3)],
    [(AugmentOp::ShearX, 0.9, 4), (AugmentOp::Solarize, 0.3, 3)],
    [(AugmentOp::ShearY, 0.8, 8), (AugmentOp::Invert, 0.7, 0)],
    [
        (AugmentOp::Equalize, 0.9, 0),
        (AugmentOp::TranslateY, 0.6, 6),
    ],
    [(AugmentOp::Invert, 0.9, 0), (AugmentOp::Equalize, 0.6, 0)],
    [(AugmentOp::Contrast, 0.3, 3), (AugmentOp::Rotate, 0.8, 4)],
    [(AugmentOp::Invert, 0.8, 0), (AugmentOp::TranslateY, 0.0, 2)],
    [(AugmentOp::ShearY, 0.7, 6), (AugmentOp::Solarize, 0.4, 8)],
    [(AugmentOp::Invert, 0.6, 0), (AugmentOp::Rotate, 0.8, 4)],
    [(AugmentOp::ShearY, 0.3, 7), (AugmentOp::TranslateX, 0.9, 3)],
    [(AugmentOp::ShearX, 0.1, 6), (AugmentOp::Invert, 0.6, 0)],
    [
        (AugmentOp::Solarize, 0.7, 2),
        (AugmentOp::TranslateY, 0.6, 7),
    ],
    [(AugmentOp::ShearY, 0.8, 4), (AugmentOp::Invert, 0.8, 0)],
    [(AugmentOp::ShearX, 0.7, 9), (AugmentOp::TranslateY, 0.8, 3)],
    [
        (AugmentOp::ShearY, 0.8, 5),
        (AugmentOp::AutoContrast, 0.7, 0),
    ],
    [(AugmentOp::ShearX, 0.7, 2), (AugmentOp::Invert, 0.1, 0)],
];

#[cfg(test)]
mod tests {
    use super::*;
    fn reference() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/torchvision_augment.json"
        ))
        .unwrap()
    }
    fn input(data: &serde_json::Value) -> ImageSample {
        let pixels: Vec<u8> = data["pixels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec(
                pixels,
                [
                    data["height"].as_u64().unwrap() as usize,
                    data["width"].as_u64().unwrap() as usize,
                    3,
                ],
                &Device::Cpu,
            )
            .unwrap(),
            label: 7,
        })
    }
    #[test]
    fn policy_steps_match_independent_torchvision_cpu_and_pil_fixtures() {
        let data = reference();
        for case in data["cases"].as_array().unwrap() {
            let step = AugmentStep {
                op: AugmentOp::parse(case["op"].as_str().unwrap()).unwrap(),
                magnitude: case["magnitude"].as_f64().unwrap() as f32,
            };
            let mode: InterpolationMode = case["interpolation"].as_str().unwrap().parse().unwrap();
            let tensor = case["backend"].as_str().unwrap() == "tensor";
            let result = if tensor {
                step.apply_backend::<true>(input(&data), mode, 37)
            } else {
                step.apply(input(&data), mode, 37)
            }
            .unwrap()
            .into_decoded()
            .unwrap();
            let values = result.image.to_vec::<u8>().unwrap();
            let max = values
                .iter()
                .zip(case["expected"].as_array().unwrap())
                .map(|(&a, b)| (i32::from(a) - b.as_i64().unwrap() as i32).abs())
                .max()
                .unwrap();
            assert!(
                max <= if tensor { 1 } else { 0 },
                "backend={tensor} op={:?} mode={mode:?} max={max}",
                step.op
            );
            assert_eq!(result.label, 7);
        }
    }
    #[test]
    fn learned_policy_tables_match_torchvision() {
        let data = reference();
        for (name, table) in [
            ("imagenet", &IMAGENET),
            ("cifar10", &CIFAR10),
            ("svhn", &SVHN),
        ] {
            let expected = data["policies"][name].as_array().unwrap();
            assert_eq!(table.len(), expected.len());
            for (pair, expected_pair) in table.iter().zip(expected) {
                for ((op, p, index), expected) in pair.iter().zip(expected_pair.as_array().unwrap())
                {
                    assert_eq!(op.name(), expected[0].as_str().unwrap());
                    assert_eq!(*p, expected[1].as_f64().unwrap() as f32);
                    assert_eq!(*index, expected[2].as_u64().unwrap_or(0) as usize);
                }
            }
        }
    }

    #[test]
    fn native_policies_vary_with_seed() {
        let data = reference();
        for kind in 0..6 {
            let mut outputs = Vec::new();
            for seed in 0..20 {
                let mut rng = RandomStream::from_seed(seed);
                let output = match kind {
                    0 => AutoAugmentConfig::new(AutoAugmentPolicy::ImageNet)
                        .apply(input(&data), &mut rng),
                    1 => AutoAugmentConfig::new(AutoAugmentPolicy::Cifar10)
                        .apply(input(&data), &mut rng),
                    2 => AutoAugmentConfig::new(AutoAugmentPolicy::Svhn)
                        .apply(input(&data), &mut rng),
                    3 => RandAugmentConfig::default().apply(input(&data), &mut rng),
                    4 => TrivialAugmentWideConfig::default().apply(input(&data), &mut rng),
                    _ => AugMixConfig::default().apply(input(&data), &mut rng),
                }
                .unwrap()
                .into_decoded()
                .unwrap()
                .image
                .to_vec::<u8>()
                .unwrap();
                outputs.push(output);
            }
            outputs.sort_unstable();
            outputs.dedup();
            assert!(outputs.len() > 1, "policy {kind} ignored its seed");
        }
    }

    #[test]
    fn magnitude_spaces_match_torchvision() {
        let data = reference();
        for space in data["spaces"].as_array().unwrap() {
            let op = AugmentOp::parse(space["op"].as_str().unwrap()).unwrap();
            let kind = space["kind"].as_str().unwrap();
            let bins = space["bins"].as_u64().unwrap() as usize;
            for (index, value) in space["magnitudes"].as_array().unwrap().iter().enumerate() {
                let (actual, signed) =
                    magnitude(op, bins, index, 9, 7, kind == "wide", kind == "mix");
                assert_eq!(signed, space["signed"].as_bool().unwrap());
                assert!(
                    (actual - value.as_f64().unwrap() as f32).abs() < 1e-5,
                    "{kind} {op:?} {index}: {actual} != {value}"
                );
            }
        }
    }
    #[test]
    fn augmix_fixed_plan_matches_tensor_reference() {
        let data = reference();
        let m = &data["mix"];
        let plan = AugMixPlan {
            original_weight: m["original_weight"].as_f64().unwrap() as f32,
            chain_weights: m["weights"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| w.as_f64().unwrap() as f32)
                .collect(),
            chains: m["chains"]
                .as_array()
                .unwrap()
                .iter()
                .map(|chain| {
                    chain
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|step| AugmentStep {
                            op: AugmentOp::parse(step[0].as_str().unwrap()).unwrap(),
                            magnitude: step[1].as_f64().unwrap() as f32,
                        })
                        .collect()
                })
                .collect(),
        };
        let result = AugMixConfig {
            fill: 37,
            ..Default::default()
        }
        .apply_plan(input(&data), &plan)
        .unwrap()
        .into_decoded()
        .unwrap()
        .image
        .to_vec::<u8>()
        .unwrap();
        let max = result
            .iter()
            .zip(m["expected"].as_array().unwrap())
            .map(|(&a, b)| (i32::from(a) - b.as_i64().unwrap() as i32).abs())
            .max()
            .unwrap();
        assert!(max <= 1, "AugMix max error {max}");
    }
    #[test]
    fn dirichlet_samples_have_expected_mean_and_variance() {
        for alpha in [0.5, 1.0, 2.0] {
            let mut rng = RandomStream::from_seed(17);
            let samples: Vec<f32> = (0..4096)
                .map(|_| {
                    let weights = dirichlet(3, alpha, &mut rng);
                    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                    assert!(weights.iter().all(|&w| w > 0.0 && w < 1.0));
                    weights[0]
                })
                .collect();
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
            assert!((mean - 1.0 / 3.0).abs() < 0.025);
            assert!((variance - 2.0 / (9.0 * (3.0 * alpha + 1.0))).abs() < 0.01);
        }
    }
}
