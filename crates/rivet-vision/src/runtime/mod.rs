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
    /// Maximum retained payload bytes on each persistent stage-graph edge.
    /// Oversized items fail explicitly instead of exceeding this bound.
    pub stage_queue_max_bytes: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_workers: 0,
            prefetch_batches: 2,
            stage_queue_max_bytes: 512 * 1024 * 1024,
        }
    }
}
