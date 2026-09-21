mod pool;
mod reorder;
mod work;

pub use pool::WorkerPool;
pub use reorder::{PendingBatch, PendingSamples, PrefetchCoordinator};
pub use work::{WorkItem, WorkResult};

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("runtime error: {0}")]
    Message(String),
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;

fn runtime_error(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Message(message.into())
}
