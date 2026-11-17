use super::context::SampleContext;
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::sample::image::ImageAxisOrder;
use crate::sample::image::ImageSample;
use crate::transforms::LinearTransformationConfig;
use crate::transforms::color::{
    AutocontrastConfig, BrightnessConfig, ColorJitterConfig, ContrastConfig, EqualizeConfig,
    GrayscaleConfig, HueConfig, InvertConfig, PosterizeConfig, RandomGrayscaleConfig,
    SharpnessConfig, SolarizeConfig,
};
use crate::transforms::geometry::{
    CenterCropConfig, CropConfig, FiveCropConfig, FlipConfig, InterpolationMode, LayoutConfig,
    PadConfig, RandomCropConfig, RandomHorizontalFlipConfig, RandomResizedCropConfig, ResizeConfig,
    RotateConfig, RotationAngle, TenCropConfig,
};
use crate::transforms::representation::{
    ConvertImageDtypeConfig, DecodeImageConfig, NormalizeConfig,
};
use crate::transforms::{
    AffineConfig, EraseConfig, GammaConfig, RandomGaussianBlurConfig, RandomRotationConfig,
    SaturationConfig,
};
use crate::transforms::{
    ArbitraryRotateConfig, ElasticTransformConfig, PerspectiveConfig, RandomAffineConfig,
    RandomPerspectiveConfig,
};
use crate::transforms::{
    AugMixConfig, AutoAugmentConfig, RandAugmentConfig, TrivialAugmentWideConfig,
};
use crate::transforms::{GaussianBlurConfig, RandomErasingConfig};
use rivet_core::DType;
use rivet_data::random::{OpKey, RandomStream};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionKind {
    Sample,
    Batch,
}

#[derive(Clone)]
pub enum ImageOp {
    Decode(DecodeImageConfig),
    LinearTransformation(LinearTransformationConfig),
    AutoAugment(AutoAugmentConfig),
    RandAugment(RandAugmentConfig),
    TrivialAugmentWide(TrivialAugmentWideConfig),
    AugMix(AugMixConfig),

    Saturation(SaturationConfig),
    Gamma(GammaConfig),
    Affine(AffineConfig),
    Erase(EraseConfig),
    RandomRotation(RandomRotationConfig),
    RandomGaussianBlur(RandomGaussianBlurConfig),

    Resize(ResizeConfig),
    Crop(CropConfig),
    CenterCrop(CenterCropConfig),
    Pad(PadConfig),
    Flip(FlipConfig),
    RandomCrop(RandomCropConfig),
    RandomResizedCrop(RandomResizedCropConfig),
    RandomHorizontalFlip(RandomHorizontalFlipConfig),
    Brightness(BrightnessConfig),
    Contrast(ContrastConfig),
    Hue(HueConfig),
    ColorJitter(ColorJitterConfig),
    Invert(InvertConfig),
    Posterize(PosterizeConfig),
    Solarize(SolarizeConfig),
    Autocontrast(AutocontrastConfig),
    Equalize(EqualizeConfig),
    Sharpness(SharpnessConfig),
    ArbitraryRotate(ArbitraryRotateConfig),
    RandomAffine(RandomAffineConfig),
    Perspective(PerspectiveConfig),
    RandomPerspective(RandomPerspectiveConfig),
    ElasticTransform(ElasticTransformConfig),
    RandomApply {
        probability: f64,
        ops: Vec<ImageOp>,
    },
    RandomChoice {
        choices: Vec<Vec<ImageOp>>,
        weights: Option<Vec<f64>>,
    },
    RandomOrder {
        ops: Vec<ImageOp>,
    },
    GaussianBlur(GaussianBlurConfig),
    Grayscale(GrayscaleConfig),
    RandomGrayscale(RandomGrayscaleConfig),
    RandomErasing(RandomErasingConfig),
    ConvertImageDtype(ConvertImageDtypeConfig),
    Rotate(RotateConfig),
    Normalize(NormalizeConfig),
    Layout(LayoutConfig),
    FiveCrop(FiveCropConfig),
    TenCrop(TenCropConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineImageState {
    Encoded,
    Decoded {
        dtype: DType,
        axis_order: ImageAxisOrder,
    },
}

impl ImageOp {
    /// Stable semantic name used to assign per-operator random namespaces.
    pub fn random_key_kind(&self) -> Option<&'static str> {
        match self {
            Self::RandomRotation(_) => Some("RandomRotation"),
            Self::RandomGaussianBlur(_) => Some("RandomGaussianBlur"),
            Self::AutoAugment(_) => Some("AutoAugment"),
            Self::RandAugment(_) => Some("RandAugment"),
            Self::TrivialAugmentWide(_) => Some("TrivialAugmentWide"),
            Self::AugMix(_) => Some("AugMix"),
            Self::RandomCrop(_) => Some("RandomCrop"),
            Self::RandomResizedCrop(_) => Some("RandomResizedCrop"),
            Self::RandomHorizontalFlip(_) => Some("RandomHorizontalFlip"),
            Self::ColorJitter(_) => Some("ColorJitter"),
            Self::RandomAffine(_) => Some("RandomAffine"),
            Self::RandomPerspective(_) => Some("RandomPerspective"),
            Self::ElasticTransform(_) => Some("ElasticTransform"),
            Self::RandomApply { .. } => Some("RandomApply"),
            Self::RandomChoice { .. } => Some("RandomChoice"),
            Self::RandomOrder { .. } => Some("RandomOrder"),
            Self::RandomGrayscale(_) => Some("RandomGrayscale"),
            Self::RandomErasing(_) => Some("RandomErasing"),
            _ => None,
        }
    }

    pub fn execution_kind(&self) -> ExecutionKind {
        match self {
            Self::Normalize(_) | Self::ConvertImageDtype(_) | Self::Layout(_) => {
                ExecutionKind::Batch
            }
            Self::Decode(_)
            | Self::LinearTransformation(_)
            | Self::AutoAugment(_)
            | Self::RandAugment(_)
            | Self::TrivialAugmentWide(_)
            | Self::AugMix(_)
            | Self::Saturation(_)
            | Self::Gamma(_)
            | Self::Affine(_)
            | Self::Erase(_)
            | Self::RandomRotation(_)
            | Self::RandomGaussianBlur(_)
            | Self::Resize(_)
            | Self::Crop(_)
            | Self::CenterCrop(_)
            | Self::FiveCrop(_)
            | Self::TenCrop(_)
            | Self::Pad(_)
            | Self::Flip(_)
            | Self::RandomCrop(_)
            | Self::RandomResizedCrop(_)
            | Self::RandomHorizontalFlip(_)
            | Self::Brightness(_)
            | Self::Contrast(_)
            | Self::Hue(_)
            | Self::ColorJitter(_)
            | Self::Invert(_)
            | Self::Posterize(_)
            | Self::Solarize(_)
            | Self::Autocontrast(_)
            | Self::Equalize(_)
            | Self::Sharpness(_)
            | Self::ArbitraryRotate(_)
            | Self::RandomAffine(_)
            | Self::Perspective(_)
            | Self::RandomPerspective(_)
            | Self::ElasticTransform(_)
            | Self::RandomApply { .. }
            | Self::RandomChoice { .. }
            | Self::RandomOrder { .. }
            | Self::GaussianBlur(_)
            | Self::Grayscale(_)
            | Self::RandomGrayscale(_)
            | Self::RandomErasing(_)
            | Self::Rotate(_) => ExecutionKind::Sample,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Saturation(_) => "Saturation",
            Self::Gamma(_) => "Gamma",
            Self::Affine(_) => "Affine",
            Self::Erase(_) => "Erase",
            Self::RandomRotation(_) => "RandomRotation",
            Self::RandomGaussianBlur(_) => "RandomGaussianBlur",
            Self::AutoAugment(_) => "AutoAugment",
            Self::RandAugment(_) => "RandAugment",
            Self::TrivialAugmentWide(_) => "TrivialAugmentWide",
            Self::AugMix(_) => "AugMix",
            Self::LinearTransformation(_) => "LinearTransformation",
            Self::Decode(_) => "Decode",
            Self::Resize(_) => "Resize",
            Self::Crop(_) => "Crop",
            Self::CenterCrop(_) => "CenterCrop",
            Self::FiveCrop(_) => "FiveCrop",
            Self::TenCrop(_) => "TenCrop",
            Self::Pad(_) => "Pad",
            Self::Flip(_) => "Flip",
            Self::RandomCrop(_) => "RandomCrop",
            Self::RandomResizedCrop(_) => "RandomResizedCrop",
            Self::RandomHorizontalFlip(_) => "RandomHorizontalFlip",
            Self::Brightness(_) => "Brightness",
            Self::Contrast(_) => "Contrast",
            Self::Hue(_) => "Hue",
            Self::ColorJitter(_) => "ColorJitter",
            Self::Invert(_) => "Invert",
            Self::Posterize(_) => "Posterize",
            Self::Solarize(_) => "Solarize",
            Self::Autocontrast(_) => "Autocontrast",
            Self::Equalize(_) => "Equalize",
            Self::Sharpness(_) => "Sharpness",
            Self::ArbitraryRotate(_) => "ArbitraryRotate",
            Self::RandomAffine(_) => "RandomAffine",
            Self::Perspective(_) => "Perspective",
            Self::RandomPerspective(_) => "RandomPerspective",
            Self::ElasticTransform(_) => "ElasticTransform",
            Self::RandomApply { .. } => "RandomApply",
            Self::RandomChoice { .. } => "RandomChoice",
            Self::RandomOrder { .. } => "RandomOrder",
            Self::GaussianBlur(_) => "GaussianBlur",
            Self::Grayscale(_) => "Grayscale",
            Self::RandomGrayscale(_) => "RandomGrayscale",
            Self::RandomErasing(_) => "RandomErasing",
            Self::ConvertImageDtype(_) => "ConvertImageDtype",
            Self::Rotate(_) => "Rotate",
            Self::Normalize(_) => "Normalize",
            Self::Layout(_) => "Layout",
        }
    }

    /// Stable semantic operation identity used by the domain planning
    /// protocol. Keep these values stable when variants are reordered.
    pub const fn planning_op_id(&self) -> u32 {
        match self {
            Self::Saturation(_) => 36,
            Self::Gamma(_) => 37,
            Self::Affine(_) => 38,
            Self::Erase(_) => 39,
            Self::RandomRotation(_) => 40,
            Self::RandomGaussianBlur(_) => 41,
            Self::AutoAugment(_) => 42,
            Self::RandAugment(_) => 43,
            Self::TrivialAugmentWide(_) => 44,
            Self::AugMix(_) => 45,
            Self::LinearTransformation(_) => 46,
            Self::FiveCrop(_) => 47,
            Self::TenCrop(_) => 48,
            Self::Decode(_) => 1,
            Self::Resize(_) => 2,
            Self::Crop(_) => 3,
            Self::CenterCrop(_) => 4,
            Self::Pad(_) => 5,
            Self::Flip(_) => 6,
            Self::RandomCrop(_) => 7,
            Self::RandomResizedCrop(_) => 8,
            Self::RandomHorizontalFlip(_) => 9,
            Self::Brightness(_) => 10,
            Self::Contrast(_) => 11,
            Self::Hue(_) => 12,
            Self::ColorJitter(_) => 13,
            Self::Invert(_) => 14,
            Self::Posterize(_) => 15,
            Self::Solarize(_) => 16,
            Self::Autocontrast(_) => 17,
            Self::Equalize(_) => 18,
            Self::Sharpness(_) => 19,
            Self::ArbitraryRotate(_) => 20,
            Self::RandomAffine(_) => 21,
            Self::Perspective(_) => 22,
            Self::RandomPerspective(_) => 23,
            Self::ElasticTransform(_) => 24,
            Self::RandomApply { .. } => 25,
            Self::RandomChoice { .. } => 26,
            Self::RandomOrder { .. } => 27,
            Self::GaussianBlur(_) => 28,
            Self::Grayscale(_) => 29,
            Self::RandomGrayscale(_) => 30,
            Self::RandomErasing(_) => 31,
            Self::ConvertImageDtype(_) => 32,
            Self::Rotate(_) => 33,
            Self::Normalize(_) => 34,
            Self::Layout(_) => 35,
        }
    }

    pub fn transition(&self, input: PipelineImageState) -> RivetResult<PipelineImageState> {
        let properties = super::super::inference::properties_from_state(input);
        let output = super::super::inference::infer_image_op(self, &properties)?;
        super::super::inference::state_from_properties(&output)
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let random_key = self
            .random_key_kind()
            .map(|kind| OpKey::from_parts(kind, 0));
        self.apply_with_state(
            sample,
            ctx,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: input_layout,
            },
            random_key,
        )
    }

    fn apply_with_state(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
        random_key: Option<OpKey>,
    ) -> RivetResult<ImageSample> {
        let input_layout = match input_state {
            PipelineImageState::Encoded => ImageAxisOrder::Hwc,
            PipelineImageState::Decoded { axis_order, .. } => axis_order,
        };
        match self {
            Self::Saturation(op) => op.apply(sample),
            Self::Gamma(op) => op.apply(sample),
            Self::Affine(op) => op.apply(sample, input_layout),
            Self::Erase(op) => op.apply(sample, input_layout),
            Self::RandomRotation(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomGaussianBlur(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::AutoAugment(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::RandAugment(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::TrivialAugmentWide(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::AugMix(op) => op.apply(sample, &mut required_rng(ctx, random_key, self.name())?),
            Self::LinearTransformation(op) => op.apply(sample, input_layout),
            Self::Decode(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample, input_layout),
            Self::CenterCrop(op) => op.apply(sample, input_layout),
            Self::FiveCrop(_) | Self::TenCrop(_) => Err(invalid_pipeline(
                "multi-view crops require the expanding pipeline executor",
            )),
            Self::Pad(op) => op.apply(sample, input_layout),
            Self::Flip(op) => op.apply_with_axis_order(sample, input_layout),
            Self::RandomCrop(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomResizedCrop(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomHorizontalFlip(op) => op.apply_with_axis_order(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
            Self::Hue(op) => op.apply(sample),
            Self::ColorJitter(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::Invert(op) => op.apply(sample, input_layout),
            Self::Posterize(op) => op.apply(sample, input_layout),
            Self::Solarize(op) => op.apply(sample, input_layout),
            Self::Autocontrast(op) => op.apply(sample, input_layout),
            Self::Equalize(op) => op.apply(sample, input_layout),
            Self::Sharpness(op) => op.apply(sample, input_layout),
            Self::ArbitraryRotate(op) => op.apply(sample, input_layout),
            Self::RandomAffine(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::Perspective(op) => op.apply(sample, input_layout),
            Self::RandomPerspective(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::ElasticTransform(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomApply { probability, ops } => {
                let mut rng = required_rng(ctx, random_key, self.name())?;
                if !rng.gen_bool(*probability)? {
                    Ok(sample)
                } else {
                    apply_sequence(sample, ops, ctx, input_state, random_key.unwrap())
                        .map(|(sample, _)| sample)
                }
            }
            Self::RandomChoice { choices, weights } => {
                let mut rng = required_rng(ctx, random_key, self.name())?;
                let index = if choices.is_empty() {
                    return Err(invalid_argument(
                        "random_choice requires at least one choice",
                    ));
                } else {
                    let cumulative = choice_cumulative_weights(choices.len(), weights.as_deref())?;
                    choose_branch(&mut rng, choices.len(), cumulative.as_deref())?
                };
                apply_sequence(
                    sample,
                    &choices[index],
                    ctx,
                    input_state,
                    random_key
                        .unwrap()
                        .derive(OpKey::from_parts("RandomChoiceBranch", index as u32)),
                )
                .map(|(sample, _)| sample)
            }
            Self::RandomOrder { ops } => {
                let mut order: Vec<usize> = (0..ops.len()).collect();
                let mut rng = required_rng(ctx, random_key, self.name())?;
                for index in (1..order.len()).rev() {
                    let swap = rng.gen_range_usize(0..index + 1)?;
                    order.swap(index, swap);
                }
                apply_ordered_sequence(sample, ops, &order, ctx, input_state, random_key.unwrap())
                    .map(|(sample, _)| sample)
            }
            Self::GaussianBlur(op) => op.apply(sample),
            Self::Grayscale(op) => op.apply(sample, input_layout),
            Self::RandomGrayscale(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomErasing(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::ConvertImageDtype(_) => Err(invalid_pipeline(
                "ConvertImageDtype is a batch-stage operation",
            )),
            Self::Rotate(op) => op.apply(sample),
            Self::Normalize(op) => op.apply(sample, input_layout),
            Self::Layout(op) => op.apply(sample, input_layout),
        }
    }

    pub fn apply_sample(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
    ) -> RivetResult<ImageSample> {
        let random_key = self
            .random_key_kind()
            .map(|kind| OpKey::from_parts(kind, 0));
        self.apply_sample_with_key(sample, ctx, input_state, random_key)
    }

    pub fn apply_sample_with_key(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
        random_key: Option<OpKey>,
    ) -> RivetResult<ImageSample> {
        if self.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{} is a batch-stage operation",
                self.name()
            )));
        }
        self.apply_with_state(sample, ctx, input_state, random_key)
    }

    /// Execute a sample-stage operation from a compiled plan.
    ///
    /// Compilation has already validated the operation kind and input state,
    /// so the runtime path can dispatch directly to the transform without
    /// repeating those semantic checks.
    pub(crate) fn apply_sample_compiled(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
        random_key: Option<OpKey>,
    ) -> RivetResult<ImageSample> {
        self.apply_with_state(sample, ctx, input_state, random_key)
    }

    pub fn apply_batch(
        &self,
        batch: rivet_core::Tensor,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<rivet_core::Tensor> {
        match self {
            Self::Normalize(op) => op.apply_batch(batch, input_layout),
            Self::ConvertImageDtype(op) => op.apply_batch(batch, input_layout),
            Self::Layout(op) => op.apply_batch(batch, input_layout),
            _ => Err(invalid_pipeline(format!(
                "{} is not a batch-stage operation",
                self.name()
            ))),
        }
    }

    pub fn decode() -> Self {
        Self::Decode(DecodeImageConfig::new())
    }

    pub fn resize(width: u32, height: u32) -> Self {
        Self::Resize(ResizeConfig::new(width, height))
    }

    pub fn resize_short_edge(
        size: u32,
        max_size: Option<u32>,
        interpolation: InterpolationMode,
    ) -> Self {
        Self::Resize(ResizeConfig::short_edge(size, max_size, interpolation))
    }

    pub fn resize_with_interpolation(
        width: u32,
        height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        Self::Resize(ResizeConfig::with_interpolation(
            width,
            height,
            interpolation,
        ))
    }

    pub fn crop(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Crop(CropConfig::new(x, y, width, height))
    }

    pub fn five_crop(width: u32, height: u32) -> Self {
        Self::FiveCrop(FiveCropConfig::new(width, height))
    }

    pub fn ten_crop(width: u32, height: u32, vertical_flip: bool) -> Self {
        Self::TenCrop(TenCropConfig::new(width, height).with_vertical_flip(vertical_flip))
    }

    /// Number of output views per input image for this operator.
    pub fn output_multiplier(&self) -> usize {
        match self {
            Self::FiveCrop(_) => 5,
            Self::TenCrop(_) => 10,
            _ => 1,
        }
    }

    pub fn center_crop(width: u32, height: u32) -> Self {
        Self::CenterCrop(CenterCropConfig::new(width, height))
    }

    pub fn pad(padding: u32) -> Self {
        Self::Pad(PadConfig::new(padding))
    }

    pub fn pad_with_fill(padding: u32, value: f32) -> Self {
        Self::Pad(PadConfig::new(padding).with_fill(value))
    }

    pub fn pad_with_sides(left: u32, top: u32, right: u32, bottom: u32, value: f32) -> Self {
        Self::Pad(PadConfig::with_sides(left, top, right, bottom).with_fill(value))
    }

    pub fn random_crop(width: u32, height: u32, padding: u32) -> Self {
        Self::RandomCrop(RandomCropConfig::new(width, height, padding))
    }

    pub fn random_crop_with_config(config: RandomCropConfig) -> Self {
        Self::RandomCrop(config)
    }

    pub fn random_resized_crop(width: u32, height: u32) -> Self {
        Self::RandomResizedCrop(RandomResizedCropConfig::new(width, height))
    }

    pub fn random_resized_crop_with_config(config: RandomResizedCropConfig) -> Self {
        Self::RandomResizedCrop(config)
    }

    pub fn horizontal_flip() -> Self {
        Self::Flip(FlipConfig::horizontal())
    }

    pub fn vertical_flip() -> Self {
        Self::Flip(FlipConfig::vertical())
    }

    pub fn random_horizontal_flip(probability: f64) -> Self {
        Self::RandomHorizontalFlip(RandomHorizontalFlipConfig::new(probability))
    }

    pub fn brightness(value: f32) -> Self {
        Self::Brightness(BrightnessConfig::new(value))
    }

    pub fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig::new(value))
    }

    pub fn hue(factor: f32) -> Self {
        Self::Hue(HueConfig::new(factor))
    }

    pub fn color_jitter(brightness: f32, contrast: f32, hue: f32) -> Self {
        Self::ColorJitter(ColorJitterConfig::new(brightness, contrast, hue))
    }

    pub fn invert() -> Self {
        Self::Invert(InvertConfig::new())
    }

    pub fn posterize(bits: u8) -> Self {
        Self::Posterize(PosterizeConfig::new(bits))
    }

    pub fn solarize(threshold: u8) -> Self {
        Self::Solarize(SolarizeConfig::new(threshold))
    }

    pub fn autocontrast() -> Self {
        Self::Autocontrast(AutocontrastConfig::new())
    }

    pub fn equalize() -> Self {
        Self::Equalize(EqualizeConfig::new())
    }

    pub fn sharpness(amount: f32) -> Self {
        Self::Sharpness(SharpnessConfig::new(amount))
    }

    pub fn arbitrary_rotate(angle: f32) -> Self {
        Self::ArbitraryRotate(ArbitraryRotateConfig::new(angle))
    }

    pub fn arbitrary_rotate_with_options(
        angle: f32,
        expand: bool,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> Self {
        Self::ArbitraryRotate(ArbitraryRotateConfig::new(angle).with_options(
            expand,
            interpolation,
            fill,
        ))
    }

    pub fn arbitrary_rotate_with_config(config: ArbitraryRotateConfig) -> Self {
        Self::ArbitraryRotate(config)
    }

    pub fn random_affine(degrees: f32) -> Self {
        Self::RandomAffine(RandomAffineConfig::new(degrees))
    }

    pub fn random_affine_with_config(config: RandomAffineConfig) -> Self {
        Self::RandomAffine(config)
    }

    pub fn perspective(
        start_points: [crate::transforms::Point2; 4],
        end_points: [crate::transforms::Point2; 4],
    ) -> Self {
        Self::Perspective(PerspectiveConfig::new(start_points, end_points))
    }

    pub fn perspective_with_config(config: PerspectiveConfig) -> Self {
        Self::Perspective(config)
    }

    pub fn random_perspective(distortion_scale: f32, probability: f64) -> Self {
        Self::RandomPerspective(RandomPerspectiveConfig::new(distortion_scale, probability))
    }

    pub fn random_perspective_with_config(config: RandomPerspectiveConfig) -> Self {
        Self::RandomPerspective(config)
    }

    pub fn elastic_transform(alpha: f32, sigma: f32) -> Self {
        Self::ElasticTransform(ElasticTransformConfig::new(alpha, sigma))
    }

    pub fn elastic_transform_with_config(config: ElasticTransformConfig) -> Self {
        Self::ElasticTransform(config)
    }

    pub fn random_apply(probability: f64, ops: Vec<Self>) -> Self {
        Self::RandomApply { probability, ops }
    }

    pub fn random_choice(choices: Vec<Vec<Self>>) -> Self {
        Self::RandomChoice {
            choices,
            weights: None,
        }
    }

    pub fn random_choice_with_weights(choices: Vec<Vec<Self>>, weights: Vec<f64>) -> Self {
        Self::RandomChoice {
            choices,
            weights: Some(weights),
        }
    }

    pub fn random_order(ops: Vec<Self>) -> Self {
        Self::RandomOrder { ops }
    }

    pub fn gaussian_blur(sigma: f32) -> Self {
        Self::GaussianBlur(GaussianBlurConfig::new(sigma))
    }

    pub fn grayscale(num_output_channels: u8) -> Self {
        Self::Grayscale(GrayscaleConfig::new(num_output_channels))
    }

    pub fn random_grayscale(probability: f64, num_output_channels: u8) -> Self {
        Self::RandomGrayscale(
            RandomGrayscaleConfig::new(probability).with_output_channels(num_output_channels),
        )
    }

    pub fn random_erasing(probability: f64) -> Self {
        Self::RandomErasing(RandomErasingConfig::new(probability))
    }

    pub fn random_erasing_with_config(config: RandomErasingConfig) -> Self {
        Self::RandomErasing(config)
    }

    pub fn convert_image_dtype(dtype: DType) -> Self {
        Self::ConvertImageDtype(ConvertImageDtypeConfig::new(dtype))
    }

    pub fn rotate(angle: RotationAngle) -> Self {
        Self::Rotate(RotateConfig::new(angle))
    }

    pub fn normalize(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self::Normalize(NormalizeConfig::new(mean, std))
    }

    pub fn hwc_to_chw() -> Self {
        Self::Layout(LayoutConfig::new(ImageAxisOrder::Chw))
    }

    pub fn chw_to_hwc() -> Self {
        Self::Layout(LayoutConfig::new(ImageAxisOrder::Hwc))
    }

    /// Validate the op's own configuration (not its position in the
    /// pipeline; that is `transition`'s job).
    pub fn validate(&self) -> RivetResult<()> {
        match self {
            Self::LinearTransformation(op) => op.validate(),
            Self::AutoAugment(op) => op.validate(),
            Self::RandAugment(op) => op.validate(),
            Self::TrivialAugmentWide(op) => op.validate(),
            Self::AugMix(op) => op.validate(),

            Self::Saturation(op) => op.validate(),
            Self::Gamma(op) => op.validate(),
            Self::Affine(op) => op.validate(),
            Self::Erase(op) => op.validate(),
            Self::RandomRotation(op) => op.validate(),
            Self::RandomGaussianBlur(op) => op.validate(),

            Self::Decode(_) | Self::Flip(_) | Self::Layout(_) => Ok(()),
            Self::Resize(op) => op.validate(),
            Self::Crop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::CenterCrop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "center_crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::FiveCrop(op) => op.validate(),
            Self::TenCrop(op) => op.validate(),
            Self::Pad(op) => op.validate(),
            Self::RandomCrop(op) => op.validate(),
            Self::RandomResizedCrop(op) => op.validate(),
            Self::RandomHorizontalFlip(op) => {
                if !(0.0..=1.0).contains(&op.probability) {
                    return Err(invalid_argument(
                        "random_horizontal_flip probability must be in [0.0, 1.0]",
                    ));
                }
                Ok(())
            }
            Self::Contrast(op) => op.validate(),
            Self::Hue(op) => op.validate(),
            Self::Brightness(op) => op.validate(),
            Self::ColorJitter(op) => op.validate(),
            Self::Invert(_) => Ok(()),
            Self::Posterize(op) => op.validate(),
            Self::Solarize(_) => Ok(()),
            Self::Autocontrast(_) | Self::Equalize(_) => Ok(()),
            Self::Sharpness(op) => op.validate(),
            Self::ArbitraryRotate(op) => op.validate(),
            Self::RandomAffine(op) => op.validate(),
            Self::Perspective(op) => op.validate(),
            Self::RandomPerspective(op) => op.validate(),
            Self::ElasticTransform(op) => op.validate(),
            Self::RandomApply { probability, ops } => {
                validate_probability(*probability, "random_apply")?;
                validate_nested_ops(ops, "random_apply")
            }
            Self::RandomChoice { choices, weights } => {
                if choices.is_empty() {
                    return Err(invalid_argument(
                        "random_choice requires at least one choice",
                    ));
                }
                choice_cumulative_weights(choices.len(), weights.as_deref())?;
                for choice in choices {
                    validate_nested_ops(choice, "random_choice")?;
                }
                Ok(())
            }
            Self::RandomOrder { ops } => validate_nested_ops(ops, "random_order"),
            Self::GaussianBlur(op) => op.validate(),
            Self::Grayscale(op) => op.validate(),
            Self::RandomGrayscale(op) => op.validate(),
            Self::RandomErasing(op) => op.validate(),
            Self::Rotate(_) => Ok(()),
            Self::ConvertImageDtype(op) => op.validate(),
            Self::Normalize(op) => op.validate(),
        }
    }
}

fn validate_probability(probability: f64, op_name: &str) -> RivetResult<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "{op_name} probability must be finite and in [0.0, 1.0]"
        )))
    }
}

fn validate_nested_ops(ops: &[ImageOp], op_name: &str) -> RivetResult<()> {
    for op in ops {
        op.validate()?;
        if op.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{op_name} nested transforms must be sample-stage operations; {} is batch-stage",
                op.name()
            )));
        }
    }
    Ok(())
}

fn apply_sequence(
    mut sample: ImageSample,
    ops: &[ImageOp],
    ctx: &SampleContext,
    mut state: PipelineImageState,
    parent_key: OpKey,
) -> RivetResult<(ImageSample, PipelineImageState)> {
    for (index, op) in ops.iter().enumerate() {
        let random_key = nested_random_key(ops, index, parent_key);
        sample = op.apply_sample_with_key(sample, ctx, state, random_key)?;
        state = op.transition(state)?;
    }
    Ok((sample, state))
}

fn apply_ordered_sequence(
    mut sample: ImageSample,
    ops: &[ImageOp],
    order: &[usize],
    ctx: &SampleContext,
    mut state: PipelineImageState,
    parent_key: OpKey,
) -> RivetResult<(ImageSample, PipelineImageState)> {
    for &index in order {
        let op = &ops[index];
        let random_key = nested_random_key(ops, index, parent_key);
        sample = op.apply_sample_with_key(sample, ctx, state, random_key)?;
        state = op.transition(state)?;
    }
    Ok((sample, state))
}

fn nested_random_key(ops: &[ImageOp], index: usize, parent_key: OpKey) -> Option<OpKey> {
    let kind = ops[index].random_key_kind()?;
    let occurrence = ops[..index]
        .iter()
        .filter(|op| op.random_key_kind() == Some(kind))
        .count() as u32;
    Some(parent_key.derive(OpKey::from_parts(kind, occurrence)))
}

fn required_rng(
    ctx: &SampleContext,
    random_key: Option<OpKey>,
    op_name: &str,
) -> RivetResult<RandomStream> {
    random_key
        .map(|key| ctx.stream(key))
        .ok_or_else(|| invalid_pipeline(format!("{op_name} is missing its random key")))
}

/// Validate relative weights and precompute a normalized cumulative distribution.
/// None preserves the existing uniform integer sampler and its random stream.
pub(crate) fn choice_cumulative_weights(
    count: usize,
    weights: Option<&[f64]>,
) -> RivetResult<Option<Vec<f64>>> {
    if count == 0 {
        return Err(invalid_argument(
            "random_choice requires at least one choice",
        ));
    }
    let Some(weights) = weights else {
        return Ok(None);
    };
    if weights.len() != count {
        return Err(invalid_argument(
            "random_choice weights must match the number of choices",
        ));
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid_argument(
            "random_choice weights must be finite and non-negative",
        ));
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid_argument(
            "random_choice weights must have a finite positive sum",
        ));
    }
    let mut sum = 0.0;
    let mut cumulative: Vec<f64> = weights
        .iter()
        .map(|weight| {
            sum += weight / total;
            sum
        })
        .collect();
    // Avoid rounding leaving a draw outside the distribution. Preserve trailing
    // zero-weight branches by saturating at the final positive weight.
    let last_positive = weights.iter().rposition(|weight| *weight > 0.0).unwrap();
    cumulative[last_positive..].fill(1.0);
    Ok(Some(cumulative))
}

pub(crate) fn choose_branch(
    rng: &mut RandomStream,
    count: usize,
    cumulative: Option<&[f64]>,
) -> RivetResult<usize> {
    match cumulative {
        None => Ok(rng.gen_range_usize(0..count)?),
        Some(cumulative) => {
            let draw = rng.next_f64();
            Ok(cumulative
                .partition_point(|&upper| upper <= draw)
                .min(count - 1))
        }
    }
}

#[cfg(test)]
mod choice_weight_tests {
    use super::choice_cumulative_weights;

    #[test]
    fn uniform_choice_keeps_original_sampler() {
        assert_eq!(choice_cumulative_weights(3, None).unwrap(), None);
    }

    #[test]
    fn normalizes_weights_and_preserves_zero_probability_branches() {
        assert_eq!(
            choice_cumulative_weights(4, Some(&[0.0, 1.0, 3.0, 0.0])).unwrap(),
            Some(vec![0.0, 0.25, 1.0, 1.0])
        );
    }

    #[test]
    fn rejects_invalid_choice_weights() {
        assert!(choice_cumulative_weights(0, None).is_err());
        for weights in [
            vec![],
            vec![1.0],
            vec![0.0, 0.0],
            vec![-1.0, 2.0],
            vec![f64::NAN, 1.0],
            vec![f64::INFINITY, 1.0],
            vec![f64::MAX, f64::MAX],
        ] {
            assert!(choice_cumulative_weights(2, Some(&weights)).is_err());
        }
    }
}
