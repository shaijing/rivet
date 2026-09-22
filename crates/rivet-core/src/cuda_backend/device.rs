use std::fmt::Display;
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaStream};

use super::module::ModuleCache;
use crate::{Error, Result};

/// CUDA execution resources owned by one logical Rivet device.
///
/// The first runtime uses one context and one default stream. Keeping the
/// stream, cuBLAS handle, and module cache here gives later storage and kernel
/// dispatch code one lifetime owner without putting runtime resources on each
/// Tensor.
#[derive(Clone, Debug)]
pub struct CudaDevice {
    ordinal: usize,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    #[allow(dead_code)]
    modules: ModuleCache,
}

impl CudaDevice {
    /// Opens the CUDA device at `ordinal` and creates its default stream and
    /// cuBLAS handle.
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal).map_err(cuda_initialization_error)?;
        let stream = context.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_initialization_error)?;

        Ok(Self {
            ordinal,
            context,
            stream,
            blas: Arc::new(blas),
            modules: ModuleCache::new(),
        })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    /// Returns the stream used by all Phase 1 CUDA work.
    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }

    pub fn cublas_handle(&self) -> Arc<CudaBlas> {
        Arc::clone(&self.blas)
    }

    /// Waits for all work enqueued on this device's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(cuda_synchronization_error)
    }
}

fn cuda_initialization_error(error: impl Display) -> Error {
    Error::CudaInitializationFailed(error.to_string())
}

fn cuda_synchronization_error(error: impl Display) -> Error {
    Error::CudaSynchronizationFailed(error.to_string())
}
