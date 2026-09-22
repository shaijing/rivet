use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};

use super::module::ModuleCache;
use super::storage::CudaStorage;
use crate::backend::BackendDevice;
use crate::{DType, Error, Result, Shape, WithDType};

/// A snapshot of the CUDA operations tracked by the first single-stream
/// runtime. The counters cover explicit Rivet-side synchronization and
/// backend transfer submissions; cudarc-internal synchronization performed by
/// an allocation destructor is intentionally outside this API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaDebugStats {
    pub synchronize_count: usize,
    pub kernel_launch_count: usize,
    pub cublas_call_count: usize,
    pub h2d_count: usize,
    pub d2h_count: usize,
    pub d2d_count: usize,
}

#[derive(Debug, Default)]
struct CudaDebugCounters {
    synchronize_count: AtomicUsize,
    kernel_launch_count: AtomicUsize,
    cublas_call_count: AtomicUsize,
    h2d_count: AtomicUsize,
    d2h_count: AtomicUsize,
    d2d_count: AtomicUsize,
}

impl CudaDebugCounters {
    fn snapshot(&self) -> CudaDebugStats {
        CudaDebugStats {
            synchronize_count: self.synchronize_count.load(Ordering::Relaxed),
            kernel_launch_count: self.kernel_launch_count.load(Ordering::Relaxed),
            cublas_call_count: self.cublas_call_count.load(Ordering::Relaxed),
            h2d_count: self.h2d_count.load(Ordering::Relaxed),
            d2h_count: self.d2h_count.load(Ordering::Relaxed),
            d2d_count: self.d2d_count.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.synchronize_count.store(0, Ordering::Relaxed);
        self.kernel_launch_count.store(0, Ordering::Relaxed);
        self.cublas_call_count.store(0, Ordering::Relaxed);
        self.h2d_count.store(0, Ordering::Relaxed);
        self.d2h_count.store(0, Ordering::Relaxed);
        self.d2d_count.store(0, Ordering::Relaxed);
    }
}

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
    modules: ModuleCache,
    debug: Arc<CudaDebugCounters>,
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
            debug: Arc::new(CudaDebugCounters::default()),
        })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    /// Returns the single default stream used by all current CUDA work.
    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }

    pub fn cublas_handle(&self) -> Arc<CudaBlas> {
        Arc::clone(&self.blas)
    }

    /// Returns a cached function from a statically compiled PTX module.
    pub fn get_or_load_func(
        &self,
        module: rivet_kernels::Module,
        function: &str,
    ) -> Result<CudaFunction> {
        self.modules
            .get_or_load_func(&self.context, module, function)
    }

    /// Returns `(loaded_modules, cached_functions)` for diagnostics and cache
    /// regression tests.
    pub fn debug_module_counts(&self) -> (usize, usize) {
        self.modules.counts()
    }

    /// Returns whether cudarc can use stream-ordered allocation/free for this
    /// CUDA context. When false, cudarc conservatively synchronizes the
    /// allocation stream before freeing a slice.
    pub fn uses_async_alloc(&self) -> bool {
        self.context.has_async_alloc()
    }

    /// Returns a snapshot of explicit CUDA synchronization and transfer
    /// activity associated with this logical device.
    pub fn debug_stats(&self) -> CudaDebugStats {
        self.debug.snapshot()
    }

    /// Clears the debug counters for a new measurement window.
    pub fn reset_debug_stats(&self) {
        self.debug.reset();
    }

    /// Waits for all work enqueued on this device's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.debug.synchronize_count.fetch_add(1, Ordering::Relaxed);
        self.stream
            .synchronize()
            .map_err(cuda_synchronization_error)
    }

    pub(crate) fn record_h2d(&self) {
        self.debug.h2d_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_d2h(&self) {
        self.debug.d2h_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_d2d(&self) {
        self.debug.d2d_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Kernel dispatch calls this immediately after successfully enqueueing a
    /// function on [`Self::cuda_stream`]. Keeping the counter here means the
    /// later module/function cache can share the same instrumentation boundary.
    #[allow(dead_code)]
    pub(crate) fn record_kernel_launch(&self) {
        self.debug
            .kernel_launch_count
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_cublas_call(&self) {
        self.debug.cublas_call_count.fetch_add(1, Ordering::Relaxed);
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
