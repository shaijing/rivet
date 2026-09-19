mod batch;
mod loader;
mod scheduler;

pub use loader::ImageDataLoader;

use crate::errors::RivetError;
use crate::sample::image::{DecodedSample, ImageSample};

pub(crate) type ImageWorkerPool =
    rivet_data::runtime::WorkerPool<ImageSample, DecodedSample, RivetError>;
pub(crate) type ImagePrefetchCoordinator = rivet_data::runtime::PrefetchCoordinator<DecodedSample>;

pub(crate) fn runtime_error(error: rivet_data::runtime::RuntimeError) -> RivetError {
    RivetError::Worker(error.to_string())
}

/// Runtime-level execution configuration, separate from batch semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// `0` keeps the synchronous inline path; `> 0` runs a persistent worker
    /// pool of that many threads. Worker count never changes sampling order.
    pub num_workers: usize,
    /// Number of *future* batches prepared ahead while the caller consumes
    /// one at a time (worker pools only); the current batch plus that many
    /// are in flight, so `0` keeps only the current batch ahead of nothing.
    pub prefetch_batches: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_workers: 0,
            prefetch_batches: 2,
        }
    }
}
