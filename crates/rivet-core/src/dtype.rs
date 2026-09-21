use crate::cpu_backend::{CpuStorage, CpuStorageRef, buffer::AlignedBuffer};
use crate::{Error, Result};
use half::{bf16, f16};

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

    fn cpu_storage_as_slice(storage: &CpuStorage) -> Result<&[Self]>;

    fn cpu_storage_ref_as_slice(storage: CpuStorageRef<'_>) -> Result<&[Self]> {
        Err(Error::UnexpectedDType {
            expected: Self::DTYPE,
            actual: storage.dtype(),
        })
    }

    fn cpu_storage_as_mut_slice(storage: &mut CpuStorage) -> Result<&mut [Self]>;
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
