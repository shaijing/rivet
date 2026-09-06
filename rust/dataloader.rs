use crate::batch::ImageBatchBuilder;
use crate::dataset::Dataset;
use crate::decoder::decode_rgb;
use crate::errors::value_err;
use crate::pipeline::{ImagePipeline, PipelineOp};
use crate::sample::{DecodedSample, EncodedImageSample, ImageBatch};
use crate::sampler::SequentialSampler;
use pyo3::prelude::*;
use std::sync::Arc;

pub(crate) struct ImageDataLoader {
    pub(crate) pipeline: Arc<ImagePipeline>,
    pub(crate) batch_size: usize,
    pub(crate) sampler: SequentialSampler,
}

impl ImageDataLoader {
    pub(crate) fn next_batch(&mut self) -> PyResult<Option<ImageBatch>> {
        let Some(indices) = self.sampler.next_indices(self.batch_size) else {
            return Ok(None);
        };

        let mut batch = ImageBatchBuilder::with_capacity(indices.len());

        for index in indices {
            let encoded = self.pipeline.dataset.get(index)?;
            let decoded = self.decode_sample(encoded)?;
            batch.push(decoded)?;
        }

        Ok(Some(batch.finish()))
    }

    fn decode_sample(&self, encoded: EncodedImageSample) -> PyResult<DecodedSample> {
        let mut decoded = None;

        for op in &self.pipeline.ops {
            match op {
                PipelineOp::DecodeImage => {
                    decoded = Some(decode_rgb(&encoded.image, encoded.label)?);
                }
                PipelineOp::Batch { .. } => {}
            }
        }

        decoded.ok_or_else(|| value_err("image must be decoded before batching"))
    }
}
