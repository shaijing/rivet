use crate::errors::RivetResult;
use crate::pipeline::op::ExecutionPlan;
use crate::sample::image::{DecodedSample, ImageBatch};
use crate::sampler::IndexSampler;

pub(super) fn finish_samples(
    plan: &ExecutionPlan,
    samples: impl IntoIterator<Item = DecodedSample>,
    capacity: usize,
) -> RivetResult<ImageBatch> {
    plan.stack_and_apply_batch_ops(samples, capacity)
}

pub(super) fn next_batch_inline(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
) -> RivetResult<Option<ImageBatch>> {
    let Some(indices) = super::scheduler::take_indices(plan, sampler)? else {
        return Ok(None);
    };

    let samples = super::scheduler::fetch_samples(plan, &indices)?;
    let samples = indices
        .into_iter()
        .zip(samples)
        .map(|(index, sample)| plan.apply_sample_ops(sample, index))
        .collect::<RivetResult<Vec<_>>>()?;

    Ok(Some(finish_samples(plan, samples, plan.batch.size)?))
}
