use crate::batch::ImageBatchBuilder;
use crate::errors::RivetResult;
use crate::pipeline::op::ExecutionPlan;
use crate::sample::ImageBatch;
use crate::sampler::IndexSampler;

pub struct ImageDataLoader {
    pub plan: ExecutionPlan,
    pub sampler: IndexSampler,
}

impl ImageDataLoader {
    pub fn next_batch(&mut self) -> RivetResult<Option<ImageBatch>> {
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
