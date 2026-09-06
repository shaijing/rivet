use crate::batch::ImageBatchBuilder;
use crate::pipeline::op::ExecutionPlan;
use crate::sample::ImageBatch;
use crate::sampler::IndexSampler;
use pyo3::prelude::*;

pub(crate) struct ImageDataLoader {
    pub(crate) plan: ExecutionPlan,
    pub(crate) sampler: IndexSampler,
}

impl ImageDataLoader {
    pub(crate) fn next_batch(&mut self) -> PyResult<Option<ImageBatch>> {
        let Some(indices) = self.sampler.next_indices(self.plan.batch.size) else {
            return Ok(None);
        };

        if self.plan.batch.drop_last && indices.len() < self.plan.batch.size {
            return Ok(None);
        }

        let mut batch = ImageBatchBuilder::with_capacity(indices.len());

        for index in indices {
            let encoded = self.plan.source.get(index)?;
            let decoded = self.plan.apply_sample_ops(encoded, index)?;
            batch.push(decoded)?;
        }

        Ok(Some(batch.finish()))
    }
}
