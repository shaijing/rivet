use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::runtime::worker::{WorkerCommand, WorkItem, WorkResult, worker_loop};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Persistent worker pool created once at loader construction and torn down
/// when the loader is dropped. The coordinator owns the sampler and submits
/// resolved indices; workers only execute samples.
///
/// The bounded work queue applies backpressure at one batch's worth of work;
/// batch-level synchronization happens in the loader (Phase 1: one batch in
/// flight at a time).
pub struct WorkerPool {
    work_tx: Sender<WorkerCommand>,
    result_rx: Receiver<WorkResult>,
    handles: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    pub fn new(
        plan: Arc<ExecutionPlan>,
        num_workers: usize,
        work_capacity: usize,
    ) -> RivetResult<Self> {
        let (work_tx, work_rx) = crossbeam_channel::bounded(work_capacity.max(1));
        let (result_tx, result_rx) = crossbeam_channel::bounded(work_capacity.max(1));

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
            work_tx,
            result_rx,
            handles,
        })
    }

    pub fn submit(&self, item: WorkItem) -> RivetResult<()> {
        self.work_tx
            .send(WorkerCommand::Run(item))
            .map_err(|_| RivetError::Worker("work queue closed".to_string()))
    }

    pub fn recv(&self) -> RivetResult<WorkResult> {
        self.result_rx
            .recv()
            .map_err(|_| RivetError::Worker("worker pool disconnected".to_string()))
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Deterministic shutdown: tell every worker to stop, then join.
        for _ in &self.handles {
            let _ = self.work_tx.send(WorkerCommand::Shutdown);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
