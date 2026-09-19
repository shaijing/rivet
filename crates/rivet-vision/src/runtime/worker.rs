use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::sample::image::{DecodedSample, ImageSample};

/// One sample-level unit of work.
///
/// `batch_id` identifies the logical batch and `position` the slot inside
/// it, so the coordinator can prefetch several batches while workers finish
/// out of order and still rebuild each batch in sampler order.
#[derive(Debug)]
pub struct WorkItem {
    pub batch_id: u64,
    pub position: usize,
    pub index: usize,
    pub sample: ImageSample,
}

pub struct WorkResult {
    pub batch_id: u64,
    pub position: usize,
    pub result: RivetResult<DecodedSample>,
}

/// The per-sample hot path. Storage has already been fetched as a logical
/// batch by the coordinator; workers only run image transformations.
pub fn execute_sample(
    plan: &ExecutionPlan,
    sample: ImageSample,
    index: usize,
) -> RivetResult<DecodedSample> {
    plan.apply_ops(sample, index)
}

/// Persistent worker loop. Workers never sample on their own: they only
/// execute already-resolved indices handed to them by the coordinator.
///
/// Termination happens purely by channel disconnect: a closed result
/// receiver (coordinator dropped) makes a blocked `send` fail, and a closed
/// work queue makes a blocked `recv` fail. No explicit shutdown message is
/// needed, which keeps drop deadlock-free even with a full result queue.
///
/// A panicking sample is caught and reported as a worker error so the
/// coordinator never waits on a dead worker forever. `catch_unwind` covers
/// only the sample hot path; protocol/channel bugs surface normally.
pub fn worker_loop(
    plan: std::sync::Arc<ExecutionPlan>,
    worker_id: usize,
    work_rx: crossbeam_channel::Receiver<WorkItem>,
    result_tx: crossbeam_channel::Sender<WorkResult>,
) {
    while let Ok(item) = work_rx.recv() {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_sample(&plan, item.sample, item.index)
        }))
        .unwrap_or_else(|_| {
            Err(RivetError::Worker(format!(
                "worker {worker_id} panicked while processing sample {}",
                item.index
            )))
        });

        let delivered = result_tx.send(WorkResult {
            batch_id: item.batch_id,
            position: item.position,
            result: outcome,
        });
        if delivered.is_err() {
            // Coordinator is gone; stop consuming.
            break;
        }
    }
}
