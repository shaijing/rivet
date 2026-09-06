use crate::dataset::{ArrowImageDatasetCore, Dataset};
use crate::errors::value_err;
use crate::image::color::{BrightnessConfig, ContrastConfig};
use crate::image::crop::{CenterCropConfig, CropConfig};
use crate::image::decode::DecodeImageConfig;
use crate::image::flip::{FlipConfig, FlipDirection};
use crate::image::layout::LayoutConfig;
use crate::image::normalize::NormalizeConfig;
use crate::image::resize::ResizeConfig;
use crate::sample::ImageLayout;
use crate::sample::{DecodedSample, EncodedImageSample, ImageSample};
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum SourceOp {
    Arrow(Arc<ArrowImageDatasetCore>),
}

impl SourceOp {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Arrow(dataset) => dataset.len(),
        }
    }

    pub(crate) fn get(&self, index: usize) -> PyResult<EncodedImageSample> {
        match self {
            Self::Arrow(dataset) => dataset.get(index),
        }
    }
}

#[derive(Clone)]
pub(crate) enum IndexOp {
    Skip { count: usize },
    Take { count: usize },
}

impl IndexOp {
    pub(crate) fn apply(&self, indices: &mut Vec<usize>) {
        match self {
            Self::Skip { count } => {
                let drain_end = (*count).min(indices.len());
                indices.drain(..drain_end);
            }
            Self::Take { count } => {
                indices.truncate(*count);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) enum SampleOp {
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
    pub(crate) fn apply(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
    ) -> PyResult<ImageSample> {
        let _sample_index = ctx.sample_index;

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
pub(crate) struct BatchConfig {
    pub(crate) size: usize,
    pub(crate) drop_last: bool,
}

impl BatchConfig {
    pub(crate) fn new(size: usize, drop_last: bool) -> PyResult<Self> {
        if size == 0 {
            return Err(value_err("batch size must be greater than 0"));
        }

        Ok(Self { size, drop_last })
    }
}

pub(crate) struct SampleContext {
    pub(crate) sample_index: usize,
}

impl SampleContext {
    pub(crate) fn new(sample_index: usize) -> Self {
        Self { sample_index }
    }
}

#[derive(Clone)]
pub(crate) struct ExecutionPlan {
    pub(crate) source: SourceOp,
    pub(crate) index_ops: Vec<IndexOp>,
    pub(crate) sample_ops: Vec<SampleOp>,
    pub(crate) batch: BatchConfig,
}

impl ExecutionPlan {
    pub(crate) fn indices(&self, len: usize) -> Vec<usize> {
        let mut indices = (0..len).collect::<Vec<_>>();

        for op in &self.index_ops {
            op.apply(&mut indices);
        }

        indices
    }

    pub(crate) fn apply_sample_ops(
        &self,
        sample: EncodedImageSample,
        sample_index: usize,
    ) -> PyResult<DecodedSample> {
        let mut ctx = SampleContext::new(sample_index);
        let mut sample = ImageSample::Encoded(sample);

        for op in &self.sample_ops {
            sample = op.apply(sample, &mut ctx)?;
        }

        sample.into_decoded()
    }
}

impl SampleOp {
    pub(crate) fn decode_image() -> Self {
        Self::DecodeImage(DecodeImageConfig)
    }

    pub(crate) fn resize(width: u32, height: u32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(value_err("resize width and height must be greater than 0"));
        }
        Ok(Self::Resize(ResizeConfig { width, height }))
    }

    pub(crate) fn crop(x: u32, y: u32, width: u32, height: u32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(value_err("crop width and height must be greater than 0"));
        }
        Ok(Self::Crop(CropConfig {
            x,
            y,
            width,
            height,
        }))
    }

    pub(crate) fn center_crop(width: u32, height: u32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(value_err(
                "center_crop width and height must be greater than 0",
            ));
        }
        Ok(Self::CenterCrop(CenterCropConfig { width, height }))
    }

    pub(crate) fn horizontal_flip() -> Self {
        Self::Flip(FlipConfig {
            direction: FlipDirection::Horizontal,
        })
    }

    pub(crate) fn vertical_flip() -> Self {
        Self::Flip(FlipConfig {
            direction: FlipDirection::Vertical,
        })
    }

    pub(crate) fn brightness(value: i32) -> Self {
        Self::Brightness(BrightnessConfig { value })
    }

    pub(crate) fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig { value })
    }

    pub(crate) fn normalize(mean: Vec<f32>, std: Vec<f32>) -> PyResult<Self> {
        Ok(Self::Normalize(NormalizeConfig::new(mean, std)?))
    }

    pub(crate) fn hwc_to_chw() -> Self {
        Self::Layout(LayoutConfig {
            layout: ImageLayout::Chw,
        })
    }

    pub(crate) fn chw_to_hwc() -> Self {
        Self::Layout(LayoutConfig {
            layout: ImageLayout::Hwc,
        })
    }
}
