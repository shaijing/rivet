use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;

/// One sample-level unit of work submitted by a modality-specific coordinator.
///
/// `batch_id` identifies the logical batch and `position` the slot inside it,
/// so callers can prefetch several batches while workers finish out of order
/// and still rebuild each batch in sampler order.
pub struct WorkItem<T> {
    pub batch_id: u64,
    pub position: usize,
    pub sample_index: usize,
    pub payload: T,
}

/// Result for one [`WorkItem`]. The task error remains owned by the caller;
/// the runtime only transports it and does not know modality semantics.
pub struct WorkResult<R, E> {
    pub batch_id: u64,
    pub position: usize,
    pub result: Result<R, E>,
}

pub(crate) fn worker_loop<T, R, E, F, P>(
    worker_id: usize,
    work_rx: Receiver<WorkItem<T>>,
    result_tx: Sender<WorkResult<R, E>>,
    execute: Arc<F>,
    panic_error: Arc<P>,
) where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
    F: Fn(T, usize) -> Result<R, E> + Send + Sync + 'static,
    P: Fn(usize, usize) -> E + Send + Sync + 'static,
{
    while let Ok(item) = work_rx.recv() {
        let batch_id = item.batch_id;
        let position = item.position;
        let sample_index = item.sample_index;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(item.payload, sample_index)
        }))
        .unwrap_or_else(|_| Err(panic_error(worker_id, sample_index)));

        let delivered = result_tx.send(WorkResult {
            batch_id,
            position,
            result: outcome,
        });
        if delivered.is_err() {
            // Coordinator is gone; stop consuming.
            break;
        }
    }
}
