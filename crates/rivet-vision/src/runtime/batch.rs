use crate::batch::ImageBatchBuilder;
use crate::errors::RivetResult;
use crate::pipeline::op::ExecutionPlan;
use crate::sample::image::{DecodedSample, ImageBatch};
use crate::sampler::IndexSampler;

pub(super) fn finish_samples(
    samples: impl IntoIterator<Item = DecodedSample>,
    capacity: usize,
) -> RivetResult<ImageBatch> {
    let mut builder = ImageBatchBuilder::with_capacity(capacity);
    for sample in samples {
        builder.push(sample)?;
    }
    builder.finish()
}

pub(super) fn next_batch_inline(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
) -> RivetResult<Option<ImageBatch>> {
    let Some(indices) = super::scheduler::take_indices(plan, sampler)? else {
        return Ok(None);
    };

    let samples = super::scheduler::fetch_samples(plan, &indices)?;
    let mut builder = ImageBatchBuilder::with_capacity(indices.len());
    for (index, sample) in indices.into_iter().zip(samples) {
        builder.push(plan.apply_ops(sample, index)?)?;
    }

    Ok(Some(builder.finish()?))
}
