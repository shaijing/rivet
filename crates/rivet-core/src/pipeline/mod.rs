pub mod op;

use crate::dataset::ArrowImageDatasetCore;
use crate::errors::{RivetResult, invalid_pipeline};
use crate::pipeline::op::{
    BatchConfig, ExecutionPlan, IndexOp, SampleOp, SourceOp, compile_sampler,
};
use crate::runtime::ImageDataLoader;
use crate::sampler::IndexSampler;
use std::sync::Arc;

#[derive(Clone)]
pub struct ImagePipeline {
    pub source: SourceOp,
    pub index_ops: Vec<IndexOp>,
    pub sample_ops: Vec<SampleOp>,
    pub batch: Option<BatchConfig>,
}

impl ImagePipeline {
    pub fn new(dataset: Arc<ArrowImageDatasetCore>) -> Self {
        Self {
            source: SourceOp::Arrow(dataset),
            index_ops: Vec::new(),
            sample_ops: Vec::new(),
            batch: None,
        }
    }

    pub fn decode_image(mut self) -> Self {
        self.sample_ops.push(SampleOp::decode_image());
        self
    }

    pub fn resize(mut self, width: u32, height: u32) -> RivetResult<Self> {
        self.sample_ops.push(SampleOp::resize(width, height)?);
        Ok(self)
    }

    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> RivetResult<Self> {
        self.sample_ops.push(SampleOp::crop(x, y, width, height)?);
        Ok(self)
    }

    pub fn center_crop(mut self, width: u32, height: u32) -> RivetResult<Self> {
        self.sample_ops.push(SampleOp::center_crop(width, height)?);
        Ok(self)
    }

    pub fn horizontal_flip(mut self) -> Self {
        self.sample_ops.push(SampleOp::horizontal_flip());
        self
    }

    pub fn vertical_flip(mut self) -> Self {
        self.sample_ops.push(SampleOp::vertical_flip());
        self
    }

    pub fn brightness(mut self, value: i32) -> Self {
        self.sample_ops.push(SampleOp::brightness(value));
        self
    }

    pub fn contrast(mut self, value: f32) -> Self {
        self.sample_ops.push(SampleOp::contrast(value));
        self
    }

    pub fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> RivetResult<Self> {
        self.sample_ops.push(SampleOp::normalize(mean, std)?);
        Ok(self)
    }

    pub fn hwc_to_chw(mut self) -> Self {
        self.sample_ops.push(SampleOp::hwc_to_chw());
        self
    }

    pub fn chw_to_hwc(mut self) -> Self {
        self.sample_ops.push(SampleOp::chw_to_hwc());
        self
    }

    pub fn skip(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Skip { count });
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Take { count });
        self
    }

    pub fn batch(mut self, size: usize, drop_last: bool) -> RivetResult<Self> {
        self.batch = Some(BatchConfig::new(size, drop_last)?);
        Ok(self)
    }

    pub fn compile(self) -> RivetResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub fn compile_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        if self.sample_ops.is_empty() {
            return Err(invalid_pipeline("pipeline requires at least one sample op"));
        }

        let batch = self
            .batch
            .ok_or_else(|| invalid_pipeline("pipeline requires .batch(size)"))?;
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
