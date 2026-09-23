mod pipeline;
mod pool;
mod reorder;
mod stage_queue;
mod stages;
mod work;

pub use pipeline::{PhysicalPipelineAdapter, PhysicalPipelineExecutor, PipelineError};
pub use pool::WorkerPool;
pub use reorder::{PendingBatch, PendingSamples, PrefetchCoordinator};
pub use stage_queue::{BoundedStageQueue, StageMessage, StageQueueLimits, StageQueueSnapshot};
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
