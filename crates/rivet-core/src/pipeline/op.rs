use crate::dataset::{ArrowImageDatasetCore, Dataset};
use crate::errors::{RivetResult, invalid_argument};
use crate::image::color::{BrightnessConfig, ContrastConfig};
use crate::image::crop::{CenterCropConfig, CropConfig};
use crate::image::decode::DecodeImageConfig;
use crate::image::flip::{FlipConfig, FlipDirection};
use crate::image::layout::LayoutConfig;
use crate::image::normalize::NormalizeConfig;
use crate::image::resize::ResizeConfig;
use crate::sample::ImageLayout;
use crate::sample::{DecodedSample, EncodedImageSample, ImageSample};
use crate::sampler::SamplerPlan;
use std::sync::Arc;

#[derive(Clone)]
pub enum SourceOp {
    Arrow(Arc<ArrowImageDatasetCore>),
}

impl SourceOp {
    pub fn len(&self) -> usize {
        match self {
            Self::Arrow(dataset) => dataset.len(),
        }
    }

    pub fn get(&self, index: usize) -> RivetResult<EncodedImageSample> {
        match self {
            Self::Arrow(dataset) => dataset.get(index),
        }
    }
}

#[derive(Clone)]
pub enum IndexOp {
    Skip { count: usize },
    Take { count: usize },
}

impl IndexOp {
    pub fn apply_range(&self, start: &mut usize, end: &mut usize) {
        match self {
            Self::Skip { count } => {
                *start = (*start + *count).min(*end);
            }
            Self::Take { count } => {
                *end = (*start + *count).min(*end);
            }
        }
    }
}

#[derive(Clone)]
pub enum SampleOp {
    DecodeImage(DecodeImageConfig),
    Resize(ResizeConfig),
    Crop(CropConfig),
    CenterCrop(CenterCropConfig),
    Flip(FlipConfig),
    Brightness(BrightnessConfig),
    Contrast(ContrastConfig),
    Normalize(NormalizeConfig),
    Layout(LayoutConfig),
}

impl SampleOp {
    pub fn apply(&self, sample: ImageSample, ctx: &mut SampleContext) -> RivetResult<ImageSample> {
        let _sample_index = ctx.sample_index;
        let _sample_seed = ctx.sample_seed();

        match self {
            Self::DecodeImage(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample),
            Self::CenterCrop(op) => op.apply(sample),
            Self::Flip(op) => op.apply(sample),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
            Self::Normalize(op) => op.apply(sample),
            Self::Layout(op) => op.apply(sample),
        }
    }
}

#[derive(Clone, Copy)]
pub struct BatchConfig {
    pub size: usize,
    pub drop_last: bool,
}

impl BatchConfig {
    pub fn new(size: usize, drop_last: bool) -> RivetResult<Self> {
        if size == 0 {
            return Err(invalid_argument("batch size must be greater than 0"));
        }

        Ok(Self { size, drop_last })
    }
}

pub struct SampleContext {
    pub sample_index: usize,
    pub epoch: u64,
    pub global_seed: u64,
}

impl SampleContext {
    pub fn new(sample_index: usize) -> Self {
        Self {
            sample_index,
            epoch: 0,
            global_seed: 0,
        }
    }

    pub fn sample_seed(&self) -> u64 {
        let mut seed = self.global_seed ^ 0x9E37_79B9_7F4A_7C15;
        seed = mix_seed(seed ^ self.epoch);
        mix_seed(seed ^ self.sample_index as u64)
    }
}

#[derive(Clone)]
pub struct ExecutionPlan {
    pub source: SourceOp,
    pub sampler: SamplerPlan,
    pub sample_ops: Vec<SampleOp>,
    pub batch: BatchConfig,
}

impl ExecutionPlan {
    pub fn apply_sample_ops(
        &self,
        sample: EncodedImageSample,
        sample_index: usize,
    ) -> RivetResult<DecodedSample> {
        let mut ctx = SampleContext::new(sample_index);
        let mut sample = ImageSample::Encoded(sample);

        for op in &self.sample_ops {
            sample = op.apply(sample, &mut ctx)?;
        }

        sample.into_decoded()
    }
}

pub fn compile_sampler(len: usize, index_ops: &[IndexOp]) -> SamplerPlan {
    let mut start = 0usize;
    let mut end = len;

    for op in index_ops {
        op.apply_range(&mut start, &mut end);
    }

    SamplerPlan::Sequential { start, end }
}

fn mix_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

impl SampleOp {
    pub fn decode_image() -> Self {
        Self::DecodeImage(DecodeImageConfig)
    }

    pub fn resize(width: u32, height: u32) -> RivetResult<Self> {
        if width == 0 || height == 0 {
            return Err(invalid_argument(
                "resize width and height must be greater than 0",
            ));
        }
        Ok(Self::Resize(ResizeConfig { width, height }))
    }

    pub fn crop(x: u32, y: u32, width: u32, height: u32) -> RivetResult<Self> {
        if width == 0 || height == 0 {
            return Err(invalid_argument(
                "crop width and height must be greater than 0",
            ));
        }
        Ok(Self::Crop(CropConfig {
            x,
            y,
            width,
            height,
        }))
    }

    pub fn center_crop(width: u32, height: u32) -> RivetResult<Self> {
        if width == 0 || height == 0 {
            return Err(invalid_argument(
                "center_crop width and height must be greater than 0",
            ));
        }
        Ok(Self::CenterCrop(CenterCropConfig { width, height }))
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

    pub fn brightness(value: i32) -> Self {
        Self::Brightness(BrightnessConfig { value })
    }

    pub fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig { value })
    }

    pub fn normalize(mean: Vec<f32>, std: Vec<f32>) -> RivetResult<Self> {
        Ok(Self::Normalize(NormalizeConfig::new(mean, std)?))
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
}
