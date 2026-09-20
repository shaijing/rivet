use super::context::SampleContext;
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::sample::image::ImageAxisOrder;
use crate::sample::image::ImageSample;
use crate::transforms::color::{
    AutocontrastConfig, BrightnessConfig, ColorJitterConfig, ContrastConfig, EqualizeConfig,
    GrayscaleConfig, HueConfig, InvertConfig, PosterizeConfig, RandomGrayscaleConfig,
    SharpnessConfig, SolarizeConfig,
};
use crate::transforms::geometry::{
    CenterCropConfig, CropConfig, FlipConfig, InterpolationMode, LayoutConfig, PadConfig,
    RandomCropConfig, RandomHorizontalFlipConfig, RandomResizedCropConfig, ResizeConfig,
    RotateConfig, RotationAngle,
};
use crate::transforms::representation::{
    ConvertImageDtypeConfig, DecodeImageConfig, NormalizeConfig,
};
use crate::transforms::{
    ArbitraryRotateConfig, ElasticTransformConfig, PerspectiveConfig, RandomAffineConfig,
    RandomPerspectiveConfig,
};
use crate::transforms::{GaussianBlurConfig, RandomErasingConfig};
use rivet_core::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionKind {
    Sample,
    Batch,
}

#[derive(Clone)]
pub enum ImageOp {
    Decode(DecodeImageConfig),
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
    RandomApply { probability: f64, ops: Vec<ImageOp> },
    RandomChoice { choices: Vec<Vec<ImageOp>> },
    RandomOrder { ops: Vec<ImageOp> },
    GaussianBlur(GaussianBlurConfig),
    Grayscale(GrayscaleConfig),
    RandomGrayscale(RandomGrayscaleConfig),
    RandomErasing(RandomErasingConfig),
    ConvertImageDtype(ConvertImageDtypeConfig),
    Rotate(RotateConfig),
    Normalize(NormalizeConfig),
    NormalizeToChw(NormalizeConfig),
    Layout(LayoutConfig),
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
    pub fn execution_kind(&self) -> ExecutionKind {
        match self {
            Self::Normalize(_) | Self::NormalizeToChw(_) | Self::Layout(_) => ExecutionKind::Batch,
            Self::Decode(_)
            | Self::Resize(_)
            | Self::Crop(_)
            | Self::CenterCrop(_)
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
            | Self::ConvertImageDtype(_)
            | Self::Rotate(_) => ExecutionKind::Sample,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Decode(_) => "Decode",
            Self::Resize(_) => "Resize",
            Self::Crop(_) => "Crop",
            Self::CenterCrop(_) => "CenterCrop",
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
            Self::NormalizeToChw(_) => "NormalizeToChw",
            Self::Layout(_) => "Layout",
        }
    }

    pub fn transition(&self, input: PipelineImageState) -> RivetResult<PipelineImageState> {
        use PipelineImageState::{Decoded, Encoded};

        match self {
            Self::Decode(_) => match input {
                Encoded => Ok(Decoded {
                    dtype: DType::U8,
                    axis_order: ImageAxisOrder::Hwc,
                }),
                Decoded { .. } => Err(invalid_pipeline(
                    "Decode requires an encoded image, current state is decoded",
                )),
            },
            Self::Resize(_) => require_u8_hwc(input, "Resize"),
            Self::Crop(_) => require_u8_decoded(input, "Crop"),
            Self::CenterCrop(_) => require_u8_decoded(input, "CenterCrop"),
            Self::Pad(_) => require_u8_or_f32_decoded(input, "Pad"),
            Self::Flip(_) => require_u8_decoded(input, "Flip"),
            Self::RandomCrop(_) => require_u8_hwc(input, "RandomCrop"),
            Self::RandomResizedCrop(_) => require_u8_hwc(input, "RandomResizedCrop"),
            Self::RandomHorizontalFlip(_) => require_u8_decoded(input, "RandomHorizontalFlip"),
            Self::Brightness(_) => require_u8_hwc(input, "Brightness"),
            Self::Contrast(_) => require_u8_hwc(input, "Contrast"),
            Self::Hue(_) => require_u8_hwc(input, "Hue"),
            Self::ColorJitter(_) => require_u8_hwc(input, "ColorJitter"),
            Self::Invert(_) => require_u8_decoded(input, "Invert"),
            Self::Posterize(_) => require_u8_decoded(input, "Posterize"),
            Self::Solarize(_) => require_u8_decoded(input, "Solarize"),
            Self::Autocontrast(_) => require_u8_decoded(input, "Autocontrast"),
            Self::Equalize(_) => require_u8_decoded(input, "Equalize"),
            Self::Sharpness(_) => require_u8_decoded(input, "Sharpness"),
            Self::ArbitraryRotate(_) => require_u8_decoded(input, "ArbitraryRotate"),
            Self::RandomAffine(_) => require_u8_decoded(input, "RandomAffine"),
            Self::Perspective(_) => require_u8_decoded(input, "Perspective"),
            Self::RandomPerspective(_) => require_u8_decoded(input, "RandomPerspective"),
            Self::ElasticTransform(_) => require_u8_decoded(input, "ElasticTransform"),
            Self::RandomApply { probability, ops } => {
                validate_probability(*probability, "random_apply")?;
                let output = transition_sequence(ops, input)?;
                if output != input {
                    return Err(invalid_pipeline(
                        "RandomApply nested transforms must preserve image state",
                    ));
                }
                Ok(input)
            }
            Self::RandomChoice { choices } => {
                if choices.is_empty() {
                    return Err(invalid_argument(
                        "random_choice requires at least one choice",
                    ));
                }
                let mut output = None;
                for choice in choices {
                    let choice_output = transition_sequence(choice, input)?;
                    if let Some(expected) = output {
                        if expected != choice_output {
                            return Err(invalid_pipeline(
                                "random_choice choices must produce the same image state",
                            ));
                        }
                    } else {
                        output = Some(choice_output);
                    }
                }
                Ok(output.expect("random_choice choices is non-empty"))
            }
            Self::RandomOrder { ops } => {
                let mut state = input;
                for op in ops {
                    let output = transition_sequence(std::slice::from_ref(op), input)?;
                    if output != input {
                        return Err(invalid_pipeline(
                            "RandomOrder nested transforms must preserve image state",
                        ));
                    }
                    state = output;
                }
                Ok(state)
            }
            Self::GaussianBlur(_) => require_u8_hwc(input, "GaussianBlur"),
            Self::Grayscale(_) => require_u8_decoded(input, "Grayscale"),
            Self::RandomGrayscale(_) => require_u8_decoded(input, "RandomGrayscale"),
            Self::RandomErasing(_) => require_u8_or_f32_decoded(input, "RandomErasing"),
            Self::ConvertImageDtype(op) => match input {
                Encoded => Err(invalid_pipeline(
                    "ConvertImageDtype requires a decoded image, current state is encoded",
                )),
                Decoded { axis_order, .. } => Ok(Decoded {
                    dtype: op.dtype,
                    axis_order,
                }),
            },
            Self::Rotate(_) => require_u8_hwc(input, "Rotate"),
            Self::Normalize(_) => match input {
                Encoded => Err(invalid_pipeline(
                    "Normalize requires a decoded image, current state is encoded",
                )),
                Decoded { axis_order, .. } => Ok(Decoded {
                    dtype: DType::F32,
                    axis_order,
                }),
            },
            Self::NormalizeToChw(_) => {
                require_u8_hwc(input, "NormalizeToChw")?;
                Ok(PipelineImageState::Decoded {
                    dtype: DType::F32,
                    axis_order: ImageAxisOrder::Chw,
                })
            }
            Self::Layout(op) => match input {
                Encoded => Err(invalid_pipeline(
                    "Layout requires a decoded image, current state is encoded",
                )),
                Decoded { dtype, .. } => Ok(Decoded {
                    dtype,
                    axis_order: op.axis_order,
                }),
            },
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.apply_with_state(
            sample,
            ctx,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: input_layout,
            },
        )
    }

    fn apply_with_state(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        input_state: PipelineImageState,
    ) -> RivetResult<ImageSample> {
        let input_layout = match input_state {
            PipelineImageState::Encoded => ImageAxisOrder::Hwc,
            PipelineImageState::Decoded { axis_order, .. } => axis_order,
        };
        match self {
            Self::Decode(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample, input_layout),
            Self::CenterCrop(op) => op.apply(sample, input_layout),
            Self::Pad(op) => op.apply(sample, input_layout),
            Self::Flip(op) => op.apply_with_axis_order(sample, input_layout),
            Self::RandomCrop(op) => op.apply(sample, ctx, input_layout),
            Self::RandomResizedCrop(op) => op.apply(sample, ctx, input_layout),
            Self::RandomHorizontalFlip(op) => op.apply_with_axis_order(sample, ctx, input_layout),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
            Self::Hue(op) => op.apply(sample),
            Self::ColorJitter(op) => op.apply(sample, ctx),
            Self::Invert(op) => op.apply(sample, input_layout),
            Self::Posterize(op) => op.apply(sample, input_layout),
            Self::Solarize(op) => op.apply(sample, input_layout),
            Self::Autocontrast(op) => op.apply(sample, input_layout),
            Self::Equalize(op) => op.apply(sample, input_layout),
            Self::Sharpness(op) => op.apply(sample, input_layout),
            Self::ArbitraryRotate(op) => op.apply(sample, input_layout),
            Self::RandomAffine(op) => op.apply(sample, ctx, input_layout),
            Self::Perspective(op) => op.apply(sample, input_layout),
            Self::RandomPerspective(op) => op.apply(sample, ctx, input_layout),
            Self::ElasticTransform(op) => op.apply(sample, ctx, input_layout),
            Self::RandomApply { probability, ops } => {
                if ctx.next_rng_f64() >= *probability {
                    Ok(sample)
                } else {
                    apply_sequence(sample, ops, ctx, input_state).map(|(sample, _)| sample)
                }
            }
            Self::RandomChoice { choices } => {
                let index = ctx.choose_index(choices.len()).ok_or_else(|| {
                    invalid_argument("random_choice requires at least one choice")
                })?;
                apply_sequence(sample, &choices[index], ctx, input_state).map(|(sample, _)| sample)
            }
            Self::RandomOrder { ops } => {
                let mut order: Vec<usize> = (0..ops.len()).collect();
                ctx.shuffle(&mut order);
                let mut sample = sample;
                let mut state = input_state;
                for index in order {
                    (sample, state) =
                        apply_sequence(sample, std::slice::from_ref(&ops[index]), ctx, state)?;
                }
                Ok(sample)
            }
            Self::GaussianBlur(op) => op.apply(sample),
            Self::Grayscale(op) => op.apply(sample, input_layout),
            Self::RandomGrayscale(op) => op.apply(sample, ctx, input_layout),
            Self::RandomErasing(op) => op.apply(sample, ctx, input_layout),
            Self::ConvertImageDtype(op) => op.apply(sample, input_layout),
            Self::Rotate(op) => op.apply(sample),
            Self::Normalize(op) => op.apply(sample, input_layout),
            Self::NormalizeToChw(_) => Err(invalid_pipeline(
                "NormalizeToChw is a batch-stage operation",
            )),
            Self::Layout(op) => op.apply(sample, input_layout),
        }
    }

    pub fn apply_sample(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        input_state: PipelineImageState,
    ) -> RivetResult<ImageSample> {
        if self.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{} is a batch-stage operation",
                self.name()
            )));
        }
        self.apply_with_state(sample, ctx, input_state)
    }

    pub fn apply_batch(
        &self,
        batch: rivet_core::Tensor,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<rivet_core::Tensor> {
        match self {
            Self::Normalize(op) => op.apply_batch(batch, input_layout),
            Self::NormalizeToChw(op) => {
                if input_layout != ImageAxisOrder::Hwc {
                    return Err(invalid_pipeline("NormalizeToChw requires HWC batch input"));
                }
                op.apply_batch_to_chw(batch)
            }
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

    pub fn center_crop(width: u32, height: u32) -> Self {
        Self::CenterCrop(CenterCropConfig::new(width, height))
    }

    pub fn pad(padding: u32) -> Self {
        Self::Pad(PadConfig::new(padding))
    }

    pub fn pad_with_fill(padding: u32, value: f32) -> Self {
        Self::Pad(PadConfig::new(padding).with_fill(value))
    }

    pub fn random_crop(width: u32, height: u32, padding: u32) -> Self {
        Self::RandomCrop(RandomCropConfig::new(width, height, padding))
    }

    pub fn random_resized_crop(width: u32, height: u32) -> Self {
        Self::RandomResizedCrop(RandomResizedCropConfig::new(width, height))
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

    pub fn brightness(value: i32) -> Self {
        Self::Brightness(BrightnessConfig::new(value))
    }

    pub fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig::new(value))
    }

    pub fn hue(degrees: i32) -> Self {
        Self::Hue(HueConfig::new(degrees))
    }

    pub fn color_jitter(brightness: i32, contrast: f32, hue: i32) -> Self {
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
        Self::RandomChoice { choices }
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
            Self::Decode(_) | Self::Flip(_) | Self::Brightness(_) | Self::Layout(_) => Ok(()),
            Self::Resize(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "resize width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
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
            Self::Pad(op) => op.validate(),
            Self::RandomCrop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "random_crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
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
            Self::Hue(_) => Ok(()),
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
            Self::RandomChoice { choices } => {
                if choices.is_empty() {
                    return Err(invalid_argument(
                        "random_choice requires at least one choice",
                    ));
                }
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
            Self::NormalizeToChw(op) => op.validate(),
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

fn transition_sequence(
    ops: &[ImageOp],
    mut state: PipelineImageState,
) -> RivetResult<PipelineImageState> {
    for op in ops {
        op.validate()?;
        if op.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{} nested transforms must be sample-stage operations",
                op.name()
            )));
        }
        state = op.transition(state)?;
    }
    Ok(state)
}

fn apply_sequence(
    mut sample: ImageSample,
    ops: &[ImageOp],
    ctx: &mut SampleContext,
    mut state: PipelineImageState,
) -> RivetResult<(ImageSample, PipelineImageState)> {
    for op in ops {
        sample = op.apply_sample(sample, ctx, state)?;
        state = op.transition(state)?;
    }
    Ok((sample, state))
}

fn require_u8_decoded(input: PipelineImageState, op_name: &str) -> RivetResult<PipelineImageState> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{op_name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: DType::U8, ..
        } => Ok(input),
        PipelineImageState::Decoded { dtype, axis_order } => Err(invalid_pipeline(format!(
            "{op_name} requires uint8 input, current state is {:?} {}",
            dtype,
            axis_order.as_str()
        ))),
    }
}

fn require_u8_or_f32_decoded(
    input: PipelineImageState,
    op_name: &str,
) -> RivetResult<PipelineImageState> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{op_name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: DType::U8 | DType::F32,
            ..
        } => Ok(input),
        PipelineImageState::Decoded { dtype, axis_order } => Err(invalid_pipeline(format!(
            "{op_name} requires uint8 or float32 input, current state is {:?} {}",
            dtype,
            axis_order.as_str()
        ))),
    }
}

fn require_u8_hwc(input: PipelineImageState, op_name: &str) -> RivetResult<PipelineImageState> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{op_name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: DType::U8,
            axis_order: ImageAxisOrder::Hwc,
        } => Ok(input),
        PipelineImageState::Decoded { dtype, axis_order } => Err(invalid_pipeline(format!(
            "{op_name} requires uint8 HWC input, current state is {:?} {}",
            dtype,
            axis_order.as_str()
        ))),
    }
}
