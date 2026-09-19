use super::context::SampleContext;
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::image::color::{BrightnessConfig, ContrastConfig};
use crate::image::crop::{CenterCropConfig, CropConfig, RandomCropConfig};
use crate::image::decode::DecodeImageConfig;
use crate::image::flip::{FlipConfig, FlipDirection, RandomHorizontalFlipConfig};
use crate::image::layout::LayoutConfig;
use crate::image::normalize::NormalizeConfig;
use crate::image::resize::ResizeConfig;
use crate::sample::image::ImageLayout;
use crate::sample::image::ImageSample;
use rivet_core::DType;

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
    Layout(LayoutConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineImageState {
    Encoded,
    Decoded { dtype: DType, layout: ImageLayout },
}

impl ImageOp {
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
            Self::Layout(op) => op.apply(sample, input_layout),
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
