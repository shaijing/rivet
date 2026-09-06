use crate::arrow_dataset::ArrowImageDatasetCore;
use crate::dataloader::ImageDataLoader;
use crate::dataset::Dataset;
use crate::errors::value_err;
use crate::sampler::SequentialSampler;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum PipelineOp {
    DecodeImage,
    Batch { size: usize },
}

#[derive(Clone)]
pub(crate) struct ImagePipeline {
    pub(crate) dataset: Arc<ArrowImageDatasetCore>,
    pub(crate) ops: Vec<PipelineOp>,
}

impl ImagePipeline {
    pub(crate) fn new(dataset: Arc<ArrowImageDatasetCore>) -> Self {
        Self {
            dataset,
            ops: Vec::new(),
        }
    }

    pub(crate) fn decode_image(mut self) -> Self {
        self.ops.push(PipelineOp::DecodeImage);
        self
    }

    pub(crate) fn batch(mut self, size: usize) -> PyResult<Self> {
        if size == 0 {
            return Err(value_err("batch size must be greater than 0"));
        }

        self.ops.push(PipelineOp::Batch { size });
        Ok(self)
    }

    pub(crate) fn compile(self) -> PyResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub(crate) fn compile_from(self, start: usize) -> PyResult<ImageDataLoader> {
        let mut batch_size = None;

        for op in &self.ops {
            match op {
                PipelineOp::DecodeImage => {}
                PipelineOp::Batch { size } => batch_size = Some(*size),
            }
        }

        let batch_size = batch_size.ok_or_else(|| value_err("pipeline requires .batch(size)"))?;
        let len = self.dataset.len();

        Ok(ImageDataLoader {
            pipeline: Arc::new(self),
            sampler: SequentialSampler::with_position(len, start),
            batch_size,
        })
    }
}
