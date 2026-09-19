use super::work::{WorkItem, WorkResult, worker_loop};
use super::{RuntimeResult, runtime_error};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Persistent generic worker pool.
///
/// The pool owns only dispatch and channel lifecycle. The `execute` callback
/// defines modality-specific work, while `panic_error` maps a caught worker
/// panic into the caller's error type.
pub struct WorkerPool<T, R, E> {
    work_tx: Option<Sender<WorkItem<T>>>,
    result_rx: Option<Receiver<WorkResult<R, E>>>,
    handles: Vec<JoinHandle<()>>,
}

impl<T, R, E> WorkerPool<T, R, E>
where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    pub fn new<F, P>(
        num_workers: usize,
        max_in_flight_samples: usize,
        execute: F,
        panic_error: P,
    ) -> RuntimeResult<Self>
    where
        F: Fn(T, usize) -> Result<R, E> + Send + Sync + 'static,
        P: Fn(usize, usize) -> E + Send + Sync + 'static,
    {
        let capacity = max_in_flight_samples.max(1);
        let (work_tx, work_rx) = crossbeam_channel::bounded(capacity);
        let (result_tx, result_rx) = crossbeam_channel::bounded(capacity);
        let execute = Arc::new(execute);
        let panic_error = Arc::new(panic_error);

        let mut handles = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            let execute = Arc::clone(&execute);
            let panic_error = Arc::clone(&panic_error);
            let handle = std::thread::Builder::new()
                .name(format!("rivet-worker-{worker_id}"))
                .spawn(move || worker_loop(worker_id, work_rx, result_tx, execute, panic_error))
                .map_err(|err| runtime_error(format!("failed to spawn worker: {err}")))?;
            handles.push(handle);
        }

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            handles,
        })
    }

    pub fn submit(&self, item: WorkItem<T>) -> RuntimeResult<()> {
        let sender = self
            .work_tx
            .as_ref()
            .ok_or_else(|| runtime_error("work queue closed"))?;
        sender
            .send(item)
            .map_err(|_| runtime_error("work queue closed"))
    }

    pub fn recv(&self) -> RuntimeResult<WorkResult<R, E>> {
        let receiver = self
            .result_rx
            .as_ref()
            .ok_or_else(|| runtime_error("worker pool disconnected"))?;
        receiver
            .recv()
            .map_err(|_| runtime_error("worker pool disconnected"))
    }
}

impl<T, R, E> Drop for WorkerPool<T, R, E> {
    fn drop(&mut self) {
        // Disconnect the result receiver first so any worker blocked in
        // result_tx.send() gets an error and exits instead of waiting forever.
        self.result_rx.take();

        // Then close the work queue so workers blocked in recv() exit too.
        self.work_tx.take();

        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WorkerPool;
    use crate::runtime::WorkItem;

    #[test]
    fn pool_transports_metadata_and_task_results() {
        let pool = WorkerPool::new(
            2,
            4,
            |payload: usize, index| Ok::<_, String>(payload + index),
            |worker, index| format!("worker {worker} panicked at {index}"),
        )
        .unwrap();

        pool.submit(WorkItem {
            batch_id: 7,
            position: 3,
            sample_index: 11,
            payload: 4,
        })
        .unwrap();

        let result = pool.recv().unwrap();
        assert_eq!(result.batch_id, 7);
        assert_eq!(result.position, 3);
        assert_eq!(result.result.unwrap(), 15);
    }

    #[test]
    fn pool_converts_panics_to_task_errors() {
        let pool = WorkerPool::new(
            1,
            1,
            |_payload: usize, _index| -> Result<usize, String> {
                panic!("boom");
            },
            |worker, index| format!("worker {worker} panicked while processing {index}"),
        )
        .unwrap();

        pool.submit(WorkItem {
            batch_id: 0,
            position: 0,
            sample_index: 9,
            payload: 1,
        })
        .unwrap();
        let result = pool.recv().unwrap();
        assert!(result.result.unwrap_err().contains("panicked"));
    }
}
