mod loader;
mod pool;
mod worker;

pub use loader::ImageDataLoader;

/// Runtime-level execution configuration, separate from batch semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// `0` keeps the synchronous inline path; `> 0` runs a persistent worker
    /// pool of that many threads. Worker count never changes sampling order.
    pub num_workers: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { num_workers: 0 }
    }
}
