//! CUDA runtime resources used by the optional CUDA backend.
//!
//! Phase 1 establishes runtime ownership; Phase 2 adds typed CUDA allocation
//! and keeps unsupported tensor operations explicit until their kernel phases.

mod device;
mod kernels;
mod module;
mod storage;

pub use device::{CudaDebugStats, CudaDevice};
pub(crate) use storage::cuda_error;
pub use storage::{CudaStorage, CudaStorageSlice, CudaStorageView};
