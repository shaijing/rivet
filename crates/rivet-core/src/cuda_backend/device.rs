use std::fmt::Display;
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaStream};

use super::module::ModuleCache;
use super::storage::CudaStorage;
use crate::backend::BackendDevice;
use crate::{DType, Error, Result, Shape, WithDType};

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

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        CudaStorage::zeros(Arc::new(self.clone()), dtype, shape.checked_elem_count()?)
    }

    fn ones(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        CudaStorage::ones(Arc::new(self.clone()), dtype, shape.checked_elem_count()?)
    }

    fn storage_from_vec<T: WithDType>(&self, data: Vec<T>) -> Result<Self::Storage> {
        T::into_cuda_storage(data, Arc::new(self.clone()))
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        T::into_cuda_storage(data.to_vec(), Arc::new(self.clone()))
    }
}

fn cuda_initialization_error(error: impl Display) -> Error {
    Error::CudaInitializationFailed(error.to_string())
}

fn cuda_synchronization_error(error: impl Display) -> Error {
    Error::CudaSynchronizationFailed(error.to_string())
}
