use crate::backend::BackendDevice;
use crate::cpu_backend::{buffer::AlignedBuffer, CpuDevice};
use crate::dtype::{ExactOutput, IntoCpuStorageBuffer};
use crate::{DType, Result, Shape, Storage, WithDType};

/// Logical device locations. CPU is the only implemented backend for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
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
        }
    }

    pub(crate) fn ones(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.ones(shape, dtype)?)),
        }
    }

    pub(crate) fn storage_from_vec<T: WithDType>(&self, data: Vec<T>) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.storage_from_vec(data)?)),
        }
    }

    pub(crate) fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.storage_from_slice(data)?)),
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
        }
    }

    pub(crate) fn storage_from_exact_iter<T, I>(&self, data: I) -> Result<Storage>
    where
        T: WithDType,
        I: ExactSizeIterator<Item = T>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::into_cpu_storage_iter(data)?)),
        }
    }

    pub(crate) fn storage_from_exact_try_iter<T, I>(&self, data: I) -> Result<Storage>
    where
        T: WithDType,
        I: ExactSizeIterator<Item = Result<T>>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::try_into_cpu_storage_iter(data)?)),
        }
    }

    pub(crate) fn storage_from_exact_fn<T, F>(&self, len: usize, fill: F) -> Result<Storage>
    where
        T: WithDType,
        F: FnOnce(&mut dyn FnMut(T) -> Result<()>) -> Result<()>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::into_cpu_storage_with(len, fill)?)),
        }
    }

    pub(crate) fn storage_from_exact_writer<T, F>(&self, len: usize, fill: F) -> Result<Storage>
    where
        T: WithDType,
        F: FnOnce(&mut ExactOutput<T>) -> Result<()>,
    {
        match self {
            Self::Cpu => Ok(Storage::Cpu(T::into_cpu_storage_writer(len, fill)?)),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}
