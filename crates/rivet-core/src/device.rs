use std::hash::{Hash, Hasher};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::backend::BackendDevice;
use crate::cpu_backend::{buffer::AlignedBuffer, CpuDevice};
use crate::dtype::{ExactOutput, IntoCpuStorageBuffer};
use crate::{DType, Result, Shape, Storage, WithDType};

#[cfg(feature = "cuda")]
use crate::{cuda_backend::CudaDevice, Error};

/// Logical device locations. A CUDA location identifies a physical GPU ordinal;
/// context and stream identity remain runtime details of `CudaDevice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,

    #[cfg(feature = "cuda")]
    Cuda {
        ordinal: usize,
    },
}

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,

    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaDevice>),
}

impl PartialEq for Device {
    fn eq(&self, rhs: &Self) -> bool {
        self.same_device(rhs)
    }
}

impl Eq for Device {}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.location().hash(state);
    }
}

impl Device {
    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,

            #[cfg(feature = "cuda")]
            Self::Cuda(device) => DeviceLocation::Cuda {
                ordinal: device.ordinal(),
            },
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    #[cfg(feature = "cuda")]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(Arc::new(CudaDevice::new(ordinal)?)))
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,

            #[cfg(feature = "cuda")]
            (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),

            #[cfg(feature = "cuda")]
            _ => false,
        }
    }

    /// Allocates backend storage through the logical device dispatch boundary.
    ///
    /// Tensor constructors should use these helpers instead of matching on a
    /// concrete backend and wrapping the result in `Storage` themselves. This
    /// keeps the CPU-only implementation explicit while leaving one place to
    /// add CUDA allocation dispatch later.
    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.zeros(shape, dtype)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable { op: "zeros" }),
        }
    }

    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.ones(shape, dtype)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable { op: "ones" }),
        }
    }

    pub(crate) fn storage_from_vec<T: WithDType>(&self, data: Vec<T>) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.storage_from_vec(data)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_vec",
            }),
        }
    }

    pub(crate) fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.storage_from_slice(data)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_slice",
            }),
        }
    }

    pub(crate) fn storage_from_aligned_buffer<T: IntoCpuStorageBuffer>(
        &self,
        buffer: AlignedBuffer<T>,
    ) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(
                crate::cpu_backend::CpuStorage::from_aligned_buffer(buffer),
            )),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_aligned_buffer",
            }),
        }
    }

    pub(crate) fn storage_from_exact_iter<T, I>(&self, data: I) -> Result<Storage>
    where
        T: WithDType,
        I: ExactSizeIterator<Item = T>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::into_cpu_storage_iter(data)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_exact_iter",
            }),
        }
    }

    pub(crate) fn storage_from_exact_try_iter<T, I>(&self, data: I) -> Result<Storage>
    where
        T: WithDType,
        I: ExactSizeIterator<Item = Result<T>>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::try_into_cpu_storage_iter(data)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_exact_try_iter",
            }),
        }
    }

    pub(crate) fn storage_from_exact_fn<T, F>(&self, len: usize, fill: F) -> Result<Storage>
    where
        T: WithDType,
        F: FnOnce(&mut dyn FnMut(T) -> Result<()>) -> Result<()>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::into_cpu_storage_with(len, fill)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_exact_fn",
            }),
        }
    }

    pub(crate) fn storage_from_exact_writer<T, F>(&self, len: usize, fill: F) -> Result<Storage>
    where
        T: WithDType,
        F: FnOnce(&mut ExactOutput<T>) -> Result<()>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::into_cpu_storage_writer(len, fill)?)),

            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(Error::CudaStorageUnavailable {
                op: "storage_from_exact_writer",
            }),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}
