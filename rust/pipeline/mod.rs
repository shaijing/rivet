pub(crate) mod op;

use crate::dataset::ArrowImageDatasetCore;
use crate::errors::value_err;
use crate::pipeline::op::{
    BatchConfig, ExecutionPlan, IndexOp, SampleOp, SourceOp, compile_sampler,
};
use crate::runtime::ImageDataLoader;
use crate::sampler::IndexSampler;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct ImagePipeline {
    pub(crate) source: SourceOp,
    pub(crate) index_ops: Vec<IndexOp>,
    pub(crate) sample_ops: Vec<SampleOp>,
    pub(crate) batch: Option<BatchConfig>,
}

impl ImagePipeline {
    pub(crate) fn new(dataset: Arc<ArrowImageDatasetCore>) -> Self {
        Self {
            source: SourceOp::Arrow(dataset),
            index_ops: Vec::new(),
            sample_ops: Vec::new(),
            batch: None,
        }
    }

    pub(crate) fn decode_image(mut self) -> Self {
        self.sample_ops.push(SampleOp::decode_image());
        self
    }

    pub(crate) fn resize(mut self, width: u32, height: u32) -> PyResult<Self> {
        self.sample_ops.push(SampleOp::resize(width, height)?);
        Ok(self)
    }

    pub(crate) fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> PyResult<Self> {
        self.sample_ops.push(SampleOp::crop(x, y, width, height)?);
        Ok(self)
    }

    pub(crate) fn center_crop(mut self, width: u32, height: u32) -> PyResult<Self> {
        self.sample_ops.push(SampleOp::center_crop(width, height)?);
        Ok(self)
    }

    pub(crate) fn horizontal_flip(mut self) -> Self {
        self.sample_ops.push(SampleOp::horizontal_flip());
        self
    }

    pub(crate) fn vertical_flip(mut self) -> Self {
        self.sample_ops.push(SampleOp::vertical_flip());
        self
    }

    pub(crate) fn brightness(mut self, value: i32) -> Self {
        self.sample_ops.push(SampleOp::brightness(value));
        self
    }

    pub(crate) fn contrast(mut self, value: f32) -> Self {
        self.sample_ops.push(SampleOp::contrast(value));
        self
    }

    pub(crate) fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> PyResult<Self> {
        self.sample_ops.push(SampleOp::normalize(mean, std)?);
        Ok(self)
    }

    pub(crate) fn hwc_to_chw(mut self) -> Self {
        self.sample_ops.push(SampleOp::hwc_to_chw());
        self
    }

    pub(crate) fn chw_to_hwc(mut self) -> Self {
        self.sample_ops.push(SampleOp::chw_to_hwc());
        self
    }

    pub(crate) fn skip(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Skip { count });
        self
    }

    pub(crate) fn take(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Take { count });
        self
    }

    pub(crate) fn batch(mut self, size: usize, drop_last: bool) -> PyResult<Self> {
        self.batch = Some(BatchConfig::new(size, drop_last)?);
        Ok(self)
    }

    pub(crate) fn compile(self) -> PyResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub(crate) fn compile_from(self, start: usize) -> PyResult<ImageDataLoader> {
        if self.sample_ops.is_empty() {
            return Err(value_err("pipeline requires at least one sample op"));
        }

        let batch = self
            .batch
            .ok_or_else(|| value_err("pipeline requires .batch(size)"))?;
        let len = self.source.len();
        let sampler = compile_sampler(len, &self.index_ops);
        let plan = ExecutionPlan {
            source: self.source,
            sampler,
            sample_ops: self.sample_ops,
            batch,
        };

        Ok(ImageDataLoader {
            sampler: IndexSampler::new(plan.sampler.clone(), start),
            plan,
        })
    }
}
