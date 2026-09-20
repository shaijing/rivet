use super::context::SampleContext;
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::sample::image::ImageLayout;
use crate::sample::image::ImageSample;
use crate::transforms::color::{BrightnessConfig, ContrastConfig};
use crate::transforms::crop::{CenterCropConfig, CropConfig, RandomCropConfig};
use crate::transforms::decode::DecodeImageConfig;
use crate::transforms::flip::{FlipConfig, FlipDirection, RandomHorizontalFlipConfig};
use crate::transforms::layout::LayoutConfig;
use crate::transforms::normalize::NormalizeConfig;
use crate::transforms::resize::ResizeConfig;
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
    Normalize(NormalizeConfig),
    NormalizeToChw(NormalizeConfig),
    Layout(LayoutConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineImageState {
    Encoded,
    Decoded { dtype: DType, layout: ImageLayout },
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
            | Self::Contrast(_) => ExecutionKind::Sample,
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
                    layout: ImageLayout::Hwc,
                }),
                Decoded { .. } => Err(invalid_pipeline(
                    "Decode requires an encoded image, current state is decoded",
                )),
            },
            Self::Resize(_) => require_u8_hwc(input, "Resize"),
            Self::Crop(_) => require_u8_hwc(input, "Crop"),
            Self::CenterCrop(_) => require_u8_hwc(input, "CenterCrop"),
            Self::Flip(_) => require_u8_hwc(input, "Flip"),
            Self::RandomCrop(_) => require_u8_hwc(input, "RandomCrop"),
            Self::RandomHorizontalFlip(_) => require_u8_hwc(input, "RandomHorizontalFlip"),
            Self::Brightness(_) => require_u8_hwc(input, "Brightness"),
            Self::Contrast(_) => require_u8_hwc(input, "Contrast"),
            Self::Normalize(_) => match input {
                Encoded => Err(invalid_pipeline(
                    "Normalize requires a decoded image, current state is encoded",
                )),
                Decoded { layout, .. } => Ok(Decoded {
                    dtype: DType::F32,
                    layout,
                }),
            },
            Self::NormalizeToChw(_) => {
                require_u8_hwc(input, "NormalizeToChw")?;
                Ok(PipelineImageState::Decoded {
                    dtype: DType::F32,
                    layout: ImageLayout::Chw,
                })
            }
            Self::Layout(op) => match input {
                Encoded => Err(invalid_pipeline(
                    "Layout requires a decoded image, current state is encoded",
                )),
                Decoded { dtype, .. } => Ok(Decoded {
                    dtype,
                    layout: op.layout,
                }),
            },
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        input_layout: ImageLayout,
    ) -> RivetResult<ImageSample> {
        match self {
            Self::Decode(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample, input_layout),
            Self::CenterCrop(op) => op.apply(sample, input_layout),
            Self::Flip(op) => op.apply(sample),
            Self::RandomCrop(op) => op.apply(sample, ctx, input_layout),
            Self::RandomHorizontalFlip(op) => op.apply(sample, ctx),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
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
        input_layout: ImageLayout,
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
        input_layout: ImageLayout,
    ) -> RivetResult<rivet_core::Tensor> {
        match self {
            Self::Normalize(op) => op.apply_batch(batch, input_layout),
            Self::NormalizeToChw(op) => {
                if input_layout != ImageLayout::Hwc {
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
        Self::Decode(DecodeImageConfig)
    }

    pub fn resize(width: u32, height: u32) -> Self {
        Self::Resize(ResizeConfig { width, height })
    }

    pub fn crop(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Crop(CropConfig {
            x,
            y,
            width,
            height,
        })
    }

    pub fn center_crop(width: u32, height: u32) -> Self {
        Self::CenterCrop(CenterCropConfig { width, height })
    }

    pub fn random_crop(width: u32, height: u32, padding: u32) -> Self {
        Self::RandomCrop(RandomCropConfig {
            width,
            height,
            padding,
        })
    }

    pub fn horizontal_flip() -> Self {
        Self::Flip(FlipConfig {
            direction: FlipDirection::Horizontal,
        })
    }

    pub fn vertical_flip() -> Self {
        Self::Flip(FlipConfig {
            direction: FlipDirection::Vertical,
        })
    }

    pub fn random_horizontal_flip(probability: f64) -> Self {
        Self::RandomHorizontalFlip(RandomHorizontalFlipConfig { probability })
    }

    pub fn brightness(value: i32) -> Self {
        Self::Brightness(BrightnessConfig { value })
    }

    pub fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig { value })
    }

    pub fn normalize(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self::Normalize(NormalizeConfig::new(mean, std))
    }

    pub fn hwc_to_chw() -> Self {
        Self::Layout(LayoutConfig {
            layout: ImageLayout::Chw,
        })
    }

    pub fn chw_to_hwc() -> Self {
        Self::Layout(LayoutConfig {
            layout: ImageLayout::Hwc,
        })
    }

    /// Validate the op's own configuration (not its position in the
    /// pipeline; that is `transition`'s job).
    pub fn validate(&self) -> RivetResult<()> {
        match self {
            Self::Decode(_)
            | Self::Flip(_)
            | Self::Brightness(_)
            | Self::Contrast(_)
            | Self::Layout(_) => Ok(()),
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
            Self::Normalize(op) => op.validate(),
            Self::NormalizeToChw(op) => op.validate(),
        }
    }
}

fn require_u8_hwc(input: PipelineImageState, op_name: &str) -> RivetResult<PipelineImageState> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{op_name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: DType::U8,
            layout: ImageLayout::Hwc,
        } => Ok(input),
        PipelineImageState::Decoded { dtype, layout } => Err(invalid_pipeline(format!(
            "{op_name} requires uint8 HWC input, current state is {:?} {}",
            dtype,
            layout.as_str()
        ))),
    }
}
