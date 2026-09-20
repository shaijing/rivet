use super::context::SampleContext;
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::sample::image::ImageAxisOrder;
use crate::sample::image::ImageSample;
use crate::transforms::color::{BrightnessConfig, ContrastConfig, GrayscaleConfig, HueConfig};
use crate::transforms::geometry::{
    CenterCropConfig, CropConfig, FlipConfig, InterpolationMode, LayoutConfig, RandomCropConfig,
    RandomHorizontalFlipConfig, ResizeConfig, RotateConfig, RotationAngle,
};
use crate::transforms::representation::{
    ConvertImageDtypeConfig, DecodeImageConfig, NormalizeConfig,
};
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
    Flip(FlipConfig),
    RandomCrop(RandomCropConfig),
    RandomHorizontalFlip(RandomHorizontalFlipConfig),
    Brightness(BrightnessConfig),
    Contrast(ContrastConfig),
    Hue(HueConfig),
    Grayscale(GrayscaleConfig),
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
            | Self::Flip(_)
            | Self::RandomCrop(_)
            | Self::RandomHorizontalFlip(_)
            | Self::Brightness(_)
            | Self::Contrast(_)
            | Self::Hue(_)
            | Self::Grayscale(_)
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
            Self::Flip(_) => "Flip",
            Self::RandomCrop(_) => "RandomCrop",
            Self::RandomHorizontalFlip(_) => "RandomHorizontalFlip",
            Self::Brightness(_) => "Brightness",
            Self::Contrast(_) => "Contrast",
            Self::Hue(_) => "Hue",
            Self::Grayscale(_) => "Grayscale",
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
            Self::Flip(_) => require_u8_decoded(input, "Flip"),
            Self::RandomCrop(_) => require_u8_hwc(input, "RandomCrop"),
            Self::RandomHorizontalFlip(_) => require_u8_decoded(input, "RandomHorizontalFlip"),
            Self::Brightness(_) => require_u8_hwc(input, "Brightness"),
            Self::Contrast(_) => require_u8_hwc(input, "Contrast"),
            Self::Hue(_) => require_u8_hwc(input, "Hue"),
            Self::Grayscale(_) => require_u8_decoded(input, "Grayscale"),
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
        match self {
            Self::Decode(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample, input_layout),
            Self::CenterCrop(op) => op.apply(sample, input_layout),
            Self::Flip(op) => op.apply_with_axis_order(sample, input_layout),
            Self::RandomCrop(op) => op.apply(sample, ctx, input_layout),
            Self::RandomHorizontalFlip(op) => op.apply_with_axis_order(sample, ctx, input_layout),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
            Self::Hue(op) => op.apply(sample),
            Self::Grayscale(op) => op.apply(sample, input_layout),
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
        input_layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        if self.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{} is a batch-stage operation",
                self.name()
            )));
        }
        self.apply(sample, ctx, input_layout)
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

    pub fn random_crop(width: u32, height: u32, padding: u32) -> Self {
        Self::RandomCrop(RandomCropConfig::new(width, height, padding))
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

    pub fn grayscale(num_output_channels: u8) -> Self {
        Self::Grayscale(GrayscaleConfig::new(num_output_channels))
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
            Self::RandomCrop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "random_crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::RandomHorizontalFlip(op) => {
                if !(0.0..=1.0).contains(&op.probability) {
                    return Err(invalid_argument(
                        "random_horizontal_flip probability must be in [0.0, 1.0]",
                    ));
                }
                Ok(())
            }
            Self::Contrast(op) => op.validate(),
            Self::Hue(_) | Self::Rotate(_) => Ok(()),
            Self::Grayscale(op) => op.validate(),
            Self::ConvertImageDtype(op) => op.validate(),
            Self::Normalize(op) => op.validate(),
            Self::NormalizeToChw(op) => op.validate(),
        }
    }
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
