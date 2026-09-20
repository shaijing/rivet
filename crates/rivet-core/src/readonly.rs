use crate::cpu_backend::{CpuStorage, CpuStorageRef};
use crate::{DType, Error, Result};
use memmap2::Mmap;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
enum ReadOnlyBytes {
    Owned(Arc<[u8]>),
    Mmap(Arc<Mmap>),
}

impl ReadOnlyBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mmap(bytes) => bytes.as_ref().as_ref(),
        }
    }
}

/// Immutable CPU storage backed by shared bytes.
///
/// The bytes can be an owned Arrow buffer or a file mapping. Tensor views can
/// borrow this storage without copying; operations that need mutable/output
/// storage materialize a normal [`CpuStorage`] at their operation boundary.
#[derive(Debug, Clone)]
pub struct ReadOnlyCpuStorage {
    bytes: ReadOnlyBytes,
    byte_offset: usize,
    len: usize,
    dtype: DType,
}

impl ReadOnlyCpuStorage {
    pub fn from_bytes(bytes: Arc<[u8]>, dtype: DType) -> Result<Self> {
        Self::from_source(ReadOnlyBytes::Owned(bytes), 0, dtype)
    }

    /// Creates storage over a byte range inside a shared buffer. This is the
    /// adapter intended for Arrow buffers whose logical values start at an
    /// aligned byte offset.
    pub fn from_bytes_with_offset(
        bytes: Arc<[u8]>,
        byte_offset: usize,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_source(ReadOnlyBytes::Owned(bytes), byte_offset, dtype)
    }

    pub fn from_mmap(path: impl AsRef<Path>, dtype: DType) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|_| Error::StorageOutOfBounds)?;
        // Mapping a regular file is safe; the mapping is retained by the
        // Arc-backed storage for as long as any tensor view exists.
        let mmap = unsafe { Mmap::map(&file).map_err(|_| Error::StorageOutOfBounds)? };
        Self::from_source(ReadOnlyBytes::Mmap(Arc::new(mmap)), 0, dtype)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ref(&self) -> CpuStorageRef<'_> {
        let bytes = &self.as_bytes()[..self.len * self.dtype.size_in_bytes()];
        macro_rules! typed {
            ($ty:ty, $variant:ident) => {{
                let (prefix, values, suffix) = unsafe { bytes.align_to::<$ty>() };
                debug_assert!(prefix.is_empty() && suffix.is_empty());
                CpuStorageRef::$variant(values)
            }};
        }
        match self.dtype {
            DType::U8 => CpuStorageRef::U8(bytes),
            DType::U32 => typed!(u32, U32),
            DType::I16 => typed!(i16, I16),
            DType::I32 => typed!(i32, I32),
            DType::I64 => typed!(i64, I64),
            DType::BF16 => typed!(half::bf16, BF16),
            DType::F16 => typed!(half::f16, F16),
            DType::F32 => typed!(f32, F32),
            DType::F64 => typed!(f64, F64),
        }
    }

    pub(crate) fn to_cpu_storage(&self) -> CpuStorage {
        match self.as_ref() {
            CpuStorageRef::U8(values) => CpuStorage::U8(values.to_vec()),
            CpuStorageRef::U32(values) => CpuStorage::U32(values.to_vec()),
            CpuStorageRef::I16(values) => CpuStorage::I16(values.to_vec()),
            CpuStorageRef::I32(values) => CpuStorage::I32(values.to_vec()),
            CpuStorageRef::I64(values) => CpuStorage::I64(values.to_vec()),
            CpuStorageRef::BF16(values) => CpuStorage::BF16(values.to_vec()),
            CpuStorageRef::F16(values) => CpuStorage::F16(values.to_vec()),
            CpuStorageRef::F32(values) => CpuStorage::F32(values.to_vec()),
            CpuStorageRef::F64(values) => CpuStorage::F64(values.to_vec()),
        }
    }

    fn from_source(bytes: ReadOnlyBytes, byte_offset: usize, dtype: DType) -> Result<Self> {
        let available = bytes.as_slice().len().saturating_sub(byte_offset);
        let size = dtype.size_in_bytes();
        if byte_offset > bytes.as_slice().len() || size == 0 || available % size != 0 {
            return Err(Error::StorageOutOfBounds);
        }
        let len = available / size;
        let storage = Self {
            bytes,
            byte_offset,
            len,
            dtype,
        };
        // Typed views must be aligned. File mappings are page aligned; this
        // check also protects future Arrow-buffer callers with a byte offset.
        let alignment = match dtype {
            DType::U8 => 1,
            DType::U32 | DType::I32 | DType::F32 => std::mem::align_of::<u32>(),
            DType::I16 | DType::BF16 | DType::F16 => std::mem::align_of::<u16>(),
            DType::I64 | DType::F64 => std::mem::align_of::<u64>(),
        };
        let ptr = storage.as_bytes().as_ptr();
        if ptr.align_offset(alignment) != 0 {
            return Err(Error::StorageOutOfBounds);
        }
        Ok(storage)
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes.as_slice()[self.byte_offset..]
    }
}
