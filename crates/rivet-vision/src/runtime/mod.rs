mod loader;

pub use loader::ImageDataLoader;

/// Runtime-level execution configuration, separate from batch semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// `0` runs sample transforms on the persistent CPU stage thread; `> 0`
    /// uses that many inter-sample workers. Worker count never changes order.
    pub num_workers: usize,
    /// Number of *future* batches prepared ahead while the caller consumes
    /// one at a time; the current batch plus that many may be in flight.
    pub prefetch_batches: usize,
    /// Maximum producer-reported payload weight on each persistent stage queue.
    /// Dequeued, executing and reordered values are excluded. The CPU pull fast
    /// path has no stage queues and does not enforce this limit.
    /// Legacy compatibility: also supplies the DAG single-value limit when
    /// dag_max_value_logical_bytes is None; never a shared execution budget.
    pub stage_queue_max_bytes: usize,
    /// Independent DAG single-output logical byte limit, checked after allocation.
    /// None retains the legacy stage_queue_max_bytes fallback.
    pub dag_max_value_logical_bytes: Option<usize>,
    pub dag_managed_budget_bytes: Option<usize>,
    /// Collect per-node timings and byte counts. Disabled by default to keep
    /// steady-state execution free from profiling overhead.
    pub profiling_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_workers: 0,
            prefetch_batches: 2,
            stage_queue_max_bytes: 512 * 1024 * 1024,
            dag_max_value_logical_bytes: None,
            dag_managed_budget_bytes: None,
            profiling_enabled: false,
        }
    }
}
