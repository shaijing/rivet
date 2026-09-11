use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::runtime::worker::{WorkItem, WorkResult, worker_loop};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Persistent worker pool created once at loader construction and torn down
/// when the loader is dropped. The coordinator owns the sampler and submits
/// resolved indices; workers only execute samples.
///
/// The bounded work queue applies backpressure at `max_in_flight` batches'
/// worth of work.
pub struct WorkerPool {
    work_tx: Option<Sender<WorkItem>>,
    result_rx: Option<Receiver<WorkResult>>,
    handles: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    pub fn new(
        plan: Arc<ExecutionPlan>,
        num_workers: usize,
        max_in_flight_samples: usize,
    ) -> RivetResult<Self> {
        let (work_tx, work_rx) = crossbeam_channel::bounded(max_in_flight_samples.max(1));
        let (result_tx, result_rx) = crossbeam_channel::bounded(max_in_flight_samples.max(1));

        let mut handles = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let plan = Arc::clone(&plan);
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("rivet-worker-{worker_id}"))
                .spawn(move || worker_loop(plan, worker_id, work_rx, result_tx))
                .map_err(|err| RivetError::Worker(format!("failed to spawn worker: {err}")))?;
            handles.push(handle);
        }

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            handles,
        })
    }

    pub fn submit(&self, item: WorkItem) -> RivetResult<()> {
        let sender = self
            .work_tx
            .as_ref()
            .ok_or_else(|| RivetError::Worker("work queue closed".to_string()))?;
        sender
            .send(item)
            .map_err(|_| RivetError::Worker("work queue closed".to_string()))
    }

    pub fn recv(&self) -> RivetResult<WorkResult> {
        let receiver = self
            .result_rx
            .as_ref()
            .ok_or_else(|| RivetError::Worker("worker pool disconnected".to_string()))?;
        receiver
            .recv()
            .map_err(|_| RivetError::Worker("worker pool disconnected".to_string()))
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Disconnect the result receiver first so any worker blocked in
        // result_tx.send() (e.g. a full result queue under deep prefetch)
        // gets an error and exits instead of waiting forever.
        self.result_rx.take();

        // Then close the work queue so workers blocked in recv() exit too.
        self.work_tx.take();

        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
