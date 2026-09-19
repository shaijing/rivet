use super::{ImagePrefetchCoordinator, ImageWorkerPool, runtime_error};
use crate::errors::{RivetError, RivetResult, invalid_argument};
use crate::pipeline::op::ExecutionPlan;
use crate::sample::image::{ImageBatch, ImageSample};
use crate::sampler::IndexSampler;

pub(super) fn take_indices(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
) -> RivetResult<Option<Vec<usize>>> {
    let Some(indices) = sampler.next_indices(plan.batch.size) else {
        return Ok(None);
    };

    if plan.batch.drop_last && indices.len() < plan.batch.size {
        return Ok(None);
    }

    Ok(Some(indices))
}

/// Source access runs on the coordinator so persistent backends can perform
/// one batch read. Keep the same terminal panic-to-error behavior as the
/// worker path for custom dataset implementations.
pub(super) fn fetch_samples(
    plan: &ExecutionPlan,
    indices: &[usize],
) -> RivetResult<Vec<ImageSample>> {
    let samples = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.source.get_many(indices)
    }))
    .unwrap_or_else(|_| {
        Err(RivetError::Worker(
            "dataset get_many panicked while fetching a batch".to_string(),
        ))
    })?;

    if samples.len() != indices.len() {
        return Err(RivetError::Worker(format!(
            "source returned {} samples for {} indices",
            samples.len(),
            indices.len()
        )));
    }
    Ok(samples)
}

pub(super) fn fetch_batch(
    plan: &ExecutionPlan,
    indices: &[usize],
) -> RivetResult<Option<ImageBatch>> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.source.get_batch(indices)
    })) {
        Ok(Some(batch)) => batch.map(Some),
        Ok(None) => Ok(None),
        Err(_) => Err(RivetError::Worker(
            "dataset get_batch panicked while fetching a batch".to_string(),
        )),
    }
}

pub(super) fn next_batch_workers(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
    pool: &ImageWorkerPool,
    coordinator: &mut ImagePrefetchCoordinator,
) -> RivetResult<Option<ImageBatch>> {
    loop {
        // Top up the in-flight window from the sampler.
        while coordinator.in_flight < coordinator.max_in_flight && !coordinator.closed {
            match take_indices(plan, sampler)? {
                Some(indices) => {
                    let samples = fetch_samples(plan, &indices)?;
                    coordinator
                        .submit(indices, samples, pool)
                        .map_err(runtime_error)?;
                }
                None => coordinator.closed = true,
            }
        }

        if let Some(samples) = coordinator.take_ready().map_err(runtime_error)? {
            let capacity = samples.len();
            return super::batch::finish_samples(plan, samples, capacity).map(Some);
        }

        if coordinator.closed && coordinator.in_flight == 0 {
            return Ok(None);
        }

        // Nothing deliverable yet: wait for the next worker result.
        let result = pool.recv().map_err(runtime_error)?;
        match result.result {
            Ok(sample) => coordinator
                .record(result.batch_id, result.position, sample)
                .map_err(runtime_error)?,
            Err(err) => return Err(err),
        }
    }
}

pub(super) fn validate_worker_capacity(
    plan: &ExecutionPlan,
    max_in_flight: usize,
) -> RivetResult<usize> {
    plan.batch
        .size
        .checked_mul(max_in_flight)
        .ok_or_else(|| invalid_argument("worker queue capacity overflow"))
}
