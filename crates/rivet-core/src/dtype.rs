use crate::cpu_backend::{
    CpuStorage, CpuStorageRef,
    buffer::{AlignedBuffer, AlignedBufferBuilder},
};
#[cfg(feature = "cuda")]
use crate::cuda_backend::{CudaDevice, CudaStorage};
use crate::{Error, Result};
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// The runtime element types supported by a Rivet tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    I16,
    I32,
    I64,
    BF16,
    F16,
    F32,
    F64,
}

impl DType {
    /// Stable name used by integration metadata and public bindings.
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::U32 => "uint32",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::BF16 => "bfloat16",
            Self::F16 => "float16",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::I16 | Self::BF16 | Self::F16 => 2,
            Self::I64 | Self::F64 => 8,
        }
    }

    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U32 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub const fn is_float(self) -> bool {
        !self.is_int()
    }
}

/// Maps a Rust element type to its runtime dtype and typed CPU storage.
pub trait WithDType: Copy + Send + Sync + 'static {
    const DTYPE: DType;

    fn into_cpu_storage(data: Vec<Self>) -> Result<CpuStorage>;

    fn into_cpu_storage_iter<I>(data: I) -> Result<CpuStorage>
    where
        I: ExactSizeIterator<Item = Self>;

    fn try_into_cpu_storage_iter<I>(data: I) -> Result<CpuStorage>
    where
        I: ExactSizeIterator<Item = Result<Self>>;

    fn into_cpu_storage_with<F>(len: usize, fill: F) -> Result<CpuStorage>
    where
        F: FnOnce(&mut dyn FnMut(Self) -> Result<()>) -> Result<()>,
        Self: Sized;

    fn into_cpu_storage_writer<F>(len: usize, fill: F) -> Result<CpuStorage>
    where
        F: FnOnce(&mut ExactOutput<Self>) -> Result<()>,
        Self: Sized;

    fn cpu_storage_as_slice(storage: &CpuStorage) -> Result<&[Self]>;

    fn cpu_storage_ref_as_slice(storage: CpuStorageRef<'_>) -> Result<&[Self]> {
        Err(Error::UnexpectedDType {
            expected: Self::DTYPE,
            actual: storage.dtype(),
        })
    }

    fn cpu_storage_as_mut_slice(storage: &mut CpuStorage) -> Result<&mut [Self]>;

    #[cfg(feature = "cuda")]
    fn into_cuda_storage(data: Vec<Self>, device: Arc<CudaDevice>) -> Result<CudaStorage> {
        let _ = (data, device);
        Err(Error::UnsupportedCudaOp {
            op: "storage_from_vec",
        })
    }

    #[cfg(feature = "cuda")]
    fn cuda_storage_to_vec(storage: &CudaStorage) -> Result<Vec<Self>> {
        let _ = storage;
        Err(Error::UnsupportedCudaOp {
            op: "device_to_host_copy",
        })
    }
}

/// A statically dispatched writer for a fixed-size tensor output.
///
/// The writer owns the final aligned allocation and requires callers to write
/// exactly one value per output element before it is finished.
pub struct ExactOutput<T> {
    builder: AlignedBufferBuilder<T>,
}

impl<T> ExactOutput<T> {
    fn new(len: usize) -> Result<Self> {
        Ok(Self {
            builder: AlignedBufferBuilder::new(len)?,
        })
    }

    #[inline]
    pub fn write_next(&mut self, value: T) -> Result<()> {
        self.builder.write_next(value)
    }

    fn finish(self) -> Result<AlignedBuffer<T>> {
        self.builder.finish()
    }
}

#[allow(dead_code)]
pub(crate) trait IntoCpuStorageBuffer: WithDType {
    fn into_cpu_storage_buffer(data: AlignedBuffer<Self>) -> CpuStorage;
}

macro_rules! impl_with_dtype {
    ($ty:ty, $variant:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$variant;

            fn into_cpu_storage(data: Vec<Self>) -> Result<CpuStorage> {
                crate::cpu_backend::buffer::AlignedBuffer::from_vec(data).map(CpuStorage::$variant)
            }

            fn into_cpu_storage_iter<I>(data: I) -> Result<CpuStorage>
            where
                I: ExactSizeIterator<Item = Self>,
            {
                let mut builder = AlignedBufferBuilder::new(data.len())?;
                for value in data {
                    builder.write_next(value)?;
                }
                builder.finish().map(CpuStorage::$variant)
            }

            fn try_into_cpu_storage_iter<I>(data: I) -> Result<CpuStorage>
            where
                I: ExactSizeIterator<Item = Result<Self>>,
            {
                let mut builder = AlignedBufferBuilder::new(data.len())?;
                for value in data {
                    builder.write_next(value?)?;
                }
                builder.finish().map(CpuStorage::$variant)
            }

            fn into_cpu_storage_with<F>(len: usize, fill: F) -> Result<CpuStorage>
            where
                F: FnOnce(&mut dyn FnMut(Self) -> Result<()>) -> Result<()>,
            {
                let mut builder = AlignedBufferBuilder::new(len)?;
                let mut write = |value| builder.write_next(value);
                fill(&mut write)?;
                builder.finish().map(CpuStorage::$variant)
            }

            fn into_cpu_storage_writer<F>(len: usize, fill: F) -> Result<CpuStorage>
            where
                F: FnOnce(&mut ExactOutput<Self>) -> Result<()>,
            {
                let mut output = ExactOutput::new(len)?;
                fill(&mut output)?;
                output.finish().map(CpuStorage::$variant)
            }

            fn cpu_storage_as_slice(storage: &CpuStorage) -> Result<&[Self]> {
                match storage {
                    CpuStorage::$variant(data) => Ok(data.as_slice()),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$variant,
                        actual: storage.dtype(),
                    }),
                }
            }

            fn cpu_storage_ref_as_slice(storage: CpuStorageRef<'_>) -> Result<&[Self]> {
                match storage {
                    CpuStorageRef::$variant(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$variant,
                        actual: storage.dtype(),
                    }),
                }
            }

            fn cpu_storage_as_mut_slice(storage: &mut CpuStorage) -> Result<&mut [Self]> {
                match storage {
                    CpuStorage::$variant(data) => Ok(data.as_mut_slice()),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$variant,
                        actual: storage.dtype(),
                    }),
                }
            }

            #[cfg(feature = "cuda")]
            fn into_cuda_storage(
                data: Vec<Self>,
                device: std::sync::Arc<crate::cuda_backend::CudaDevice>,
            ) -> Result<crate::cuda_backend::CudaStorage> {
                let allocation = crate::cuda_backend::CudaStorage::from_host_vec(
                    std::sync::Arc::clone(&device),
                    data,
                )?;
                Ok(crate::cuda_backend::CudaStorage::from_data(
                    device,
                    crate::cuda_backend::CudaStorageSlice::$variant(allocation),
                ))
            }

            #[cfg(feature = "cuda")]
            fn cuda_storage_to_vec(
                storage: &crate::cuda_backend::CudaStorage,
            ) -> Result<Vec<Self>> {
                match &storage.data {
                    crate::cuda_backend::CudaStorageSlice::$variant(data) => {
                        data.stream().clone_dtoh(data).map_err(|error| {
                            crate::cuda_backend::cuda_error("device to host copy", error)
                        })
                    }
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$variant,
                        actual: storage.dtype(),
                    }),
                }
            }
        }

        impl IntoCpuStorageBuffer for $ty {
            fn into_cpu_storage_buffer(data: AlignedBuffer<Self>) -> CpuStorage {
                CpuStorage::$variant(data)
            }
        }
    };
}

impl_with_dtype!(u8, U8);
impl_with_dtype!(u32, U32);
impl_with_dtype!(i16, I16);
impl_with_dtype!(i32, I32);
impl_with_dtype!(i64, I64);
impl_with_dtype!(bf16, BF16);
impl_with_dtype!(f16, F16);
impl_with_dtype!(f32, F32);
impl_with_dtype!(f64, F64);
