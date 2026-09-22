//! CUDA runtime resources used by the optional CUDA backend.
//!
//! Phase 1 only establishes device/context/stream ownership. CUDA storage and
//! tensor operation dispatch are added in later phases.

mod device;
mod module;

pub use device::CudaDevice;
