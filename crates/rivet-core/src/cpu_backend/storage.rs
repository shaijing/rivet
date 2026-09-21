use super::buffer::{AlignedBuffer, AlignedBufferBuilder};
use crate::dtype::IntoCpuStorageBuffer;
use crate::{DType, Error, Layout, Result};
use half::{bf16, f16};

pub(crate) trait IntoAlignedBuffer<T> {
    fn into_aligned(self) -> Result<AlignedBuffer<T>>;
}

impl<T: Copy> IntoAlignedBuffer<T> for Vec<T> {
    fn into_aligned(self) -> Result<AlignedBuffer<T>> {
        AlignedBuffer::from_vec(self)
    }
}

impl<T> IntoAlignedBuffer<T> for AlignedBuffer<T> {
    fn into_aligned(self) -> Result<AlignedBuffer<T>> {
        Ok(self)
    }
}

pub(crate) fn aligned<T, V: IntoAlignedBuffer<T>>(values: V) -> Result<AlignedBuffer<T>> {
    values.into_aligned()
}

// The enum remains part of the existing public storage API while its backing
// allocation stays crate-private until the aligned buffer API is stabilized.
#[allow(private_interfaces)]
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(AlignedBuffer<u8>),
    U32(AlignedBuffer<u32>),
    I16(AlignedBuffer<i16>),
    I32(AlignedBuffer<i32>),
    I64(AlignedBuffer<i64>),
    BF16(AlignedBuffer<bf16>),
    F16(AlignedBuffer<f16>),
    F32(AlignedBuffer<f32>),
    F64(AlignedBuffer<f64>),
}

#[derive(Debug, Clone, Copy)]
pub enum CpuStorageRef<'a> {
    U8(&'a [u8]),
    U32(&'a [u32]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl CpuStorageRef<'_> {
    pub fn dtype(self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

#[derive(Debug)]
pub enum CpuStorageMutRef<'a> {
    U8(&'a mut [u8]),
    U32(&'a mut [u32]),
    I16(&'a mut [i16]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
    BF16(&'a mut [bf16]),
    F16(&'a mut [f16]),
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
}

impl CpuStorage {
    #[allow(dead_code)]
    pub(crate) fn from_aligned_buffer<T: IntoCpuStorageBuffer>(buffer: AlignedBuffer<T>) -> Self {
        T::into_cpu_storage_buffer(buffer)
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::I16(data) => data.len(),
            Self::I32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::BF16(data) => data.len(),
            Self::F16(data) => data.len(),
            Self::F32(data) => data.len(),
            Self::F64(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the backing allocation base address.
    ///
    /// For an empty storage no allocation exists, so the returned dangling
    /// address must not be dereferenced. Non-empty Rivet-owned allocations
    /// satisfy [`super::buffer::CPU_STORAGE_ALIGNMENT`].
    pub fn base_ptr(&self) -> *const u8 {
        match self {
            Self::U8(data) => data.base_ptr(),
            Self::U32(data) => data.base_ptr(),
            Self::I16(data) => data.base_ptr(),
            Self::I32(data) => data.base_ptr(),
            Self::I64(data) => data.base_ptr(),
            Self::BF16(data) => data.base_ptr(),
            Self::F16(data) => data.base_ptr(),
            Self::F32(data) => data.base_ptr(),
            Self::F64(data) => data.base_ptr(),
        }
    }

    /// Returns the alignment guaranteed for the backing allocation.
    pub fn base_alignment(&self) -> usize {
        match self {
            Self::U8(data) => data.alignment(),
            Self::U32(data) => data.alignment(),
            Self::I16(data) => data.alignment(),
            Self::I32(data) => data.alignment(),
            Self::I64(data) => data.alignment(),
            Self::BF16(data) => data.alignment(),
            Self::F16(data) => data.alignment(),
            Self::F32(data) => data.alignment(),
            Self::F64(data) => data.alignment(),
        }
    }

    /// Returns the guaranteed alignment of the first logical element in a
    /// view. This can be lower than [`Self::base_alignment`] when the view has
    /// a non-zero element offset.
    pub fn effective_alignment(&self, layout: &Layout) -> Result<usize> {
        crate::storage::validate_layout_for_storage(layout, self.len())?;
        if layout.checked_elem_count()? == 0 {
            return Ok(self.base_alignment());
        }
        let byte_offset = layout
            .start_offset()
            .checked_mul(self.dtype().size_in_bytes())
            .ok_or(Error::StorageOutOfBounds)?;
        Ok(super::buffer::alignment_after_offset(
            self.base_alignment(),
            byte_offset,
        ))
    }

    pub fn as_ref(&self) -> CpuStorageRef<'_> {
        match self {
            Self::U8(data) => CpuStorageRef::U8(data.as_slice()),
            Self::U32(data) => CpuStorageRef::U32(data.as_slice()),
            Self::I16(data) => CpuStorageRef::I16(data.as_slice()),
            Self::I32(data) => CpuStorageRef::I32(data.as_slice()),
            Self::I64(data) => CpuStorageRef::I64(data.as_slice()),
            Self::BF16(data) => CpuStorageRef::BF16(data.as_slice()),
            Self::F16(data) => CpuStorageRef::F16(data.as_slice()),
            Self::F32(data) => CpuStorageRef::F32(data.as_slice()),
            Self::F64(data) => CpuStorageRef::F64(data.as_slice()),
        }
    }

    pub fn as_mut(&mut self) -> CpuStorageMutRef<'_> {
        match self {
            Self::U8(data) => CpuStorageMutRef::U8(data.as_mut_slice()),
            Self::U32(data) => CpuStorageMutRef::U32(data.as_mut_slice()),
            Self::I16(data) => CpuStorageMutRef::I16(data.as_mut_slice()),
            Self::I32(data) => CpuStorageMutRef::I32(data.as_mut_slice()),
            Self::I64(data) => CpuStorageMutRef::I64(data.as_mut_slice()),
            Self::BF16(data) => CpuStorageMutRef::BF16(data.as_mut_slice()),
            Self::F16(data) => CpuStorageMutRef::F16(data.as_mut_slice()),
            Self::F32(data) => CpuStorageMutRef::F32(data.as_mut_slice()),
            Self::F64(data) => CpuStorageMutRef::F64(data.as_mut_slice()),
        }
    }

    fn value_at(&self, index: usize) -> Result<CpuValue> {
        let value = match self {
            Self::U8(data) => CpuValue::U8(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::U32(data) => CpuValue::U32(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::I16(data) => CpuValue::I16(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::I32(data) => CpuValue::I32(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::I64(data) => CpuValue::I64(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::BF16(data) => CpuValue::BF16(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::F16(data) => CpuValue::F16(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::F32(data) => CpuValue::F32(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
            Self::F64(data) => CpuValue::F64(*data.get(index).ok_or(Error::StorageOutOfBounds)?),
        };
        Ok(value)
    }

    /// Materializes a logical layout into a new contiguous storage.
    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        crate::storage::validate_layout_for_storage(layout, self.len())?;
        macro_rules! convert {
            ($variant:ident, $ty:ty, $convert:expr) => {{
                let mut builder = AlignedBufferBuilder::new(layout.checked_elem_count()?)?;
                for index in layout.strided_index() {
                    builder.write_next($convert(self.value_at(index)?.to_f64()))?;
                }
                Ok(Self::$variant(builder.finish()?))
            }};
        }

        match dtype {
            DType::U8 => convert!(U8, u8, |value: f64| value as u8),
            DType::U32 => convert!(U32, u32, |value: f64| value as u32),
            DType::I16 => convert!(I16, i16, |value: f64| value as i16),
            DType::I32 => convert!(I32, i32, |value: f64| value as i32),
            DType::I64 => convert!(I64, i64, |value: f64| value as i64),
            DType::BF16 => convert!(BF16, bf16, bf16::from_f64),
            DType::F16 => convert!(F16, f16, f16::from_f64),
            DType::F32 => convert!(F32, f32, |value: f64| value as f32),
            DType::F64 => convert!(F64, f64, |value: f64| value),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CpuValue {
    U8(u8),
    U32(u32),
    I16(i16),
    I32(i32),
    I64(i64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
}

impl CpuValue {
    fn to_f64(self) -> f64 {
        match self {
            Self::U8(value) => value as f64,
            Self::U32(value) => value as f64,
            Self::I16(value) => value as f64,
            Self::I32(value) => value as f64,
            Self::I64(value) => value as f64,
            Self::BF16(value) => value.to_f64(),
            Self::F16(value) => value.to_f64(),
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
        }
    }
}
