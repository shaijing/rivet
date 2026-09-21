use super::buffer::AlignedBuffer;
use super::storage::CpuStorage;
use crate::backend::{BackendDevice, BackendStorage};
use crate::{DType, Layout, Result, Shape};
use half::{bf16, f16};

#[derive(Clone, Debug, Default)]
pub struct CpuDevice;

static CPU_DEVICE: CpuDevice = CpuDevice;

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn device(&self) -> &Self::Device {
        &CPU_DEVICE
    }

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        self.to_dtype(layout, dtype)
    }
}

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.checked_elem_count()?;
        Ok(match dtype {
            DType::U8 => CpuStorage::U8(AlignedBuffer::from_vec(vec![0; count])?),
            DType::U32 => CpuStorage::U32(AlignedBuffer::from_vec(vec![0; count])?),
            DType::I16 => CpuStorage::I16(AlignedBuffer::from_vec(vec![0; count])?),
            DType::I32 => CpuStorage::I32(AlignedBuffer::from_vec(vec![0; count])?),
            DType::I64 => CpuStorage::I64(AlignedBuffer::from_vec(vec![0; count])?),
            DType::BF16 => {
                CpuStorage::BF16(AlignedBuffer::from_vec(vec![bf16::from_f32(0.0); count])?)
            }
            DType::F16 => {
                CpuStorage::F16(AlignedBuffer::from_vec(vec![f16::from_f32(0.0); count])?)
            }
            DType::F32 => CpuStorage::F32(AlignedBuffer::from_vec(vec![0.0; count])?),
            DType::F64 => CpuStorage::F64(AlignedBuffer::from_vec(vec![0.0; count])?),
        })
    }

    fn ones(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.checked_elem_count()?;
        Ok(match dtype {
            DType::U8 => CpuStorage::U8(AlignedBuffer::from_vec(vec![1; count])?),
            DType::U32 => CpuStorage::U32(AlignedBuffer::from_vec(vec![1; count])?),
            DType::I16 => CpuStorage::I16(AlignedBuffer::from_vec(vec![1; count])?),
            DType::I32 => CpuStorage::I32(AlignedBuffer::from_vec(vec![1; count])?),
            DType::I64 => CpuStorage::I64(AlignedBuffer::from_vec(vec![1; count])?),
            DType::BF16 => {
                CpuStorage::BF16(AlignedBuffer::from_vec(vec![bf16::from_f32(1.0); count])?)
            }
            DType::F16 => {
                CpuStorage::F16(AlignedBuffer::from_vec(vec![f16::from_f32(1.0); count])?)
            }
            DType::F32 => CpuStorage::F32(AlignedBuffer::from_vec(vec![1.0; count])?),
            DType::F64 => CpuStorage::F64(AlignedBuffer::from_vec(vec![1.0; count])?),
        })
    }

    fn storage_from_vec<T: crate::WithDType>(&self, data: Vec<T>) -> Result<Self::Storage> {
        T::into_cpu_storage(data)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        T::into_cpu_storage(data.to_vec())
    }
}
