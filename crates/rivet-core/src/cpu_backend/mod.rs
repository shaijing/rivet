pub(crate) mod buffer;
pub(crate) mod index;
pub(crate) mod math;
pub(crate) mod matmul;
pub mod utils;

use crate::backend::{BackendDevice, BackendStorage};
use crate::dtype::IntoCpuStorageBuffer;
use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::{DType, Error, Layout, Result, Shape, WithDType};
use buffer::AlignedBuffer;
use half::{bf16, f16};
use utils::{
    arg_reduce_map, binary_map, binary_scalar_map, cat_map, cmp_map, cmp_scalar_map, copy_logical,
    mean_all_map, mean_map, reduce_all_map, reduce_map, unary_map, var_map, where_map,
};

fn aligned<T: Copy>(values: Vec<T>) -> Result<AlignedBuffer<T>> {
    AlignedBuffer::from_vec(values)
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
        macro_rules! convert {
            ($variant:ident, $ty:ty, $convert:expr) => {{
                let mut output = Vec::<$ty>::with_capacity(layout.elem_count());
                for index in layout.strided_index() {
                    output.push($convert(self.value_at(index)?.to_f64()));
                }
                Ok(Self::$variant(aligned(output)?))
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

    pub(crate) fn binary(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (self, rhs) {
                    (Self::$variant(lhs), Self::$variant(rhs)) => Ok(Self::$variant(aligned(
                        binary_map(lhs, lhs_layout, rhs, rhs_layout, op)?,
                    )?)),
                    _ => Err(Error::DTypeMismatch {
                        lhs: self.dtype(),
                        rhs: rhs.dtype(),
                    }),
                }
            };
        }

        match self.dtype() {
            DType::U8 => dispatch!(U8, u8),
            DType::U32 => dispatch!(U32, u32),
            DType::I16 => dispatch!(I16, i16),
            DType::I32 => dispatch!(I32, i32),
            DType::I64 => dispatch!(I64, i64),
            DType::BF16 => dispatch!(BF16, bf16),
            DType::F16 => dispatch!(F16, f16),
            DType::F32 => dispatch!(F32, f32),
            DType::F64 => dispatch!(F64, f64),
        }
    }

    pub(crate) fn matmul(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(aligned(matmul::f32(
                lhs, lhs_layout, rhs, rhs_layout,
            )?)?)),
            _ => Err(Error::UnsupportedMatmulDType {
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty, $convert:expr) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(math::affine_map(
                        values, layout, $convert,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8, |value: u8| {
                value.wrapping_mul(mul as u8).wrapping_add(add as u8)
            }),
            Self::U32(_) => dispatch!(U32, u32, |value: u32| {
                value.wrapping_mul(mul as u32).wrapping_add(add as u32)
            }),
            Self::I16(_) => dispatch!(I16, i16, |value: i16| {
                value.wrapping_mul(mul as i16).wrapping_add(add as i16)
            }),
            Self::I32(_) => dispatch!(I32, i32, |value: i32| {
                value.wrapping_mul(mul as i32).wrapping_add(add as i32)
            }),
            Self::I64(_) => dispatch!(I64, i64, |value: i64| {
                value.wrapping_mul(mul as i64).wrapping_add(add as i64)
            }),
            Self::BF16(_) => dispatch!(BF16, bf16, |value: bf16| {
                value * bf16::from_f64(mul) + bf16::from_f64(add)
            }),
            Self::F16(_) => dispatch!(F16, f16, |value: f16| {
                value * f16::from_f64(mul) + f16::from_f64(add)
            }),
            Self::F32(_) => dispatch!(F32, f32, |value: f32| { value * mul as f32 + add as f32 }),
            Self::F64(_) => dispatch!(F64, f64, |value: f64| value * mul + add),
        }
    }

    pub(crate) fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(math::elu_map(
                        values, layout, alpha,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
            _ => Err(Error::UnsupportedDTypeForOp {
                op: "elu",
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn powf(&self, layout: &Layout, exponent: f64) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(math::powf_map(
                        values, layout, exponent,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
            _ => Err(Error::UnsupportedDTypeForOp {
                op: "powf",
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn pow(&self, lhs_layout: &Layout, rhs: &Self, rhs_layout: &Layout) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (self, rhs) {
                    (Self::$variant(lhs), Self::$variant(rhs)) => Ok(Self::$variant(aligned(
                        math::pow_map(lhs, lhs_layout, rhs, rhs_layout)?,
                    )?)),
                    _ => unreachable!(),
                }
            };
        }

        match self {
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
            _ => Err(Error::UnsupportedDTypeForOp {
                op: "pow",
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn dot(&self, lhs_layout: &Layout, rhs: &Self, rhs_layout: &Layout) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (self, rhs) {
                    (Self::$variant(lhs), Self::$variant(rhs)) => Ok(Self::$variant(aligned(
                        math::dot_map(lhs, lhs_layout, rhs, rhs_layout)?,
                    )?)),
                    _ => unreachable!(),
                }
            };
        }

        match self {
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
            _ => Err(Error::UnsupportedDTypeForOp {
                op: "dot",
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn norm(&self, layout: &Layout) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(math::norm_map(values, layout)?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
            _ => Err(Error::UnsupportedDTypeForOp {
                op: "norm",
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn cumsum(&self, layout: &Layout, dim: usize) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(math::cumsum_map(
                        values, layout, dim,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn log_sum_exp(&self, layout: &Layout, dim: usize) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(math::log_sum_exp_map(
                        values, layout, dim,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
            _ => Err(Error::UnsupportedDTypeForOp {
                op: "log_sum_exp",
                dtype: self.dtype(),
            }),
        }
    }

    pub(crate) fn flip(&self, layout: &Layout, dims: &[usize]) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(index::flip_map(
                        values, layout, dims,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn gather(
        &self,
        values_layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        macro_rules! dispatch_indexes {
            ($values_variant:ident, $values_ty:ty, $values:expr) => {
                match indexes {
                    Self::U8(ids) => Ok(Self::$values_variant(aligned(index::gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$values_variant(aligned(index::gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$values_variant(aligned(index::gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$values_variant(aligned(index::gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$values_variant(aligned(index::gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    _ => Err(Error::UnsupportedDTypeForOp {
                        op: "gather",
                        dtype: indexes.dtype(),
                    }),
                }
            };
        }

        match self {
            Self::U8(values) => dispatch_indexes!(U8, u8, values),
            Self::U32(values) => dispatch_indexes!(U32, u32, values),
            Self::I16(values) => dispatch_indexes!(I16, i16, values),
            Self::I32(values) => dispatch_indexes!(I32, i32, values),
            Self::I64(values) => dispatch_indexes!(I64, i64, values),
            Self::BF16(values) => dispatch_indexes!(BF16, bf16, values),
            Self::F16(values) => dispatch_indexes!(F16, f16, values),
            Self::F32(values) => dispatch_indexes!(F32, f32, values),
            Self::F64(values) => dispatch_indexes!(F64, f64, values),
        }
    }

    pub(crate) fn index_select(
        &self,
        values_layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        macro_rules! dispatch_indexes {
            ($values_variant:ident, $values:expr) => {
                match indexes {
                    Self::U8(ids) => Ok(Self::$values_variant(aligned(index::index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$values_variant(aligned(index::index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$values_variant(aligned(index::index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$values_variant(aligned(index::index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$values_variant(aligned(index::index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    _ => Err(Error::UnsupportedDTypeForOp {
                        op: "index_select",
                        dtype: indexes.dtype(),
                    }),
                }
            };
        }

        match self {
            Self::U8(values) => dispatch_indexes!(U8, values),
            Self::U32(values) => dispatch_indexes!(U32, values),
            Self::I16(values) => dispatch_indexes!(I16, values),
            Self::I32(values) => dispatch_indexes!(I32, values),
            Self::I64(values) => dispatch_indexes!(I64, values),
            Self::BF16(values) => dispatch_indexes!(BF16, values),
            Self::F16(values) => dispatch_indexes!(F16, values),
            Self::F32(values) => dispatch_indexes!(F32, values),
            Self::F64(values) => dispatch_indexes!(F64, values),
        }
    }

    pub(crate) fn scatter(
        &self,
        values_layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
        add: bool,
    ) -> Result<Self> {
        if self.dtype() != source.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: source.dtype(),
            });
        }
        macro_rules! dispatch_indexes {
            ($variant:ident, $values:expr, $source:expr) => {
                match indexes {
                    Self::U8(ids) => Ok(Self::$variant(aligned(index::scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$variant(aligned(index::scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$variant(aligned(index::scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$variant(aligned(index::scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$variant(aligned(index::scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    _ => Err(Error::UnsupportedDTypeForOp {
                        op: if add { "scatter_add" } else { "scatter" },
                        dtype: indexes.dtype(),
                    }),
                }
            };
        }

        match (self, source) {
            (Self::U8(values), Self::U8(source)) => {
                dispatch_indexes!(U8, values, source)
            }
            (Self::U32(values), Self::U32(source)) => {
                dispatch_indexes!(U32, values, source)
            }
            (Self::I16(values), Self::I16(source)) => {
                dispatch_indexes!(I16, values, source)
            }
            (Self::I32(values), Self::I32(source)) => {
                dispatch_indexes!(I32, values, source)
            }
            (Self::I64(values), Self::I64(source)) => {
                dispatch_indexes!(I64, values, source)
            }
            (Self::BF16(values), Self::BF16(source)) => {
                dispatch_indexes!(BF16, values, source)
            }
            (Self::F16(values), Self::F16(source)) => {
                dispatch_indexes!(F16, values, source)
            }
            (Self::F32(values), Self::F32(source)) => {
                dispatch_indexes!(F32, values, source)
            }
            (Self::F64(values), Self::F64(source)) => {
                dispatch_indexes!(F64, values, source)
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn index_add(
        &self,
        values_layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype() != source.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: source.dtype(),
            });
        }
        macro_rules! dispatch_indexes {
            ($variant:ident, $values:expr, $source:expr) => {
                match indexes {
                    Self::U8(ids) => Ok(Self::$variant(aligned(index::index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$variant(aligned(index::index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$variant(aligned(index::index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$variant(aligned(index::index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$variant(aligned(index::index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    _ => Err(Error::UnsupportedDTypeForOp {
                        op: "index_add",
                        dtype: indexes.dtype(),
                    }),
                }
            };
        }

        match (self, source) {
            (Self::U8(values), Self::U8(source)) => {
                dispatch_indexes!(U8, values, source)
            }
            (Self::U32(values), Self::U32(source)) => {
                dispatch_indexes!(U32, values, source)
            }
            (Self::I16(values), Self::I16(source)) => {
                dispatch_indexes!(I16, values, source)
            }
            (Self::I32(values), Self::I32(source)) => {
                dispatch_indexes!(I32, values, source)
            }
            (Self::I64(values), Self::I64(source)) => {
                dispatch_indexes!(I64, values, source)
            }
            (Self::BF16(values), Self::BF16(source)) => {
                dispatch_indexes!(BF16, values, source)
            }
            (Self::F16(values), Self::F16(source)) => {
                dispatch_indexes!(F16, values, source)
            }
            (Self::F32(values), Self::F32(source)) => {
                dispatch_indexes!(F32, values, source)
            }
            (Self::F64(values), Self::F64(source)) => {
                dispatch_indexes!(F64, values, source)
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn copy_logical(&self, layout: &Layout) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(copy_logical(values, layout)?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn binary_scalar<T: WithDType>(
        &self,
        layout: &Layout,
        scalar: T,
        op: BinaryOp,
    ) -> Result<Self> {
        let scalar = T::into_cpu_storage(vec![scalar])?;
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (self, scalar) {
                    (Self::$variant(values), Self::$variant(scalar)) => Ok(Self::$variant(
                        aligned(binary_scalar_map(values, layout, scalar.as_slice()[0], op)?)?,
                    )),
                    _ => Err(Error::DTypeMismatch {
                        lhs: self.dtype(),
                        rhs: T::DTYPE,
                    }),
                }
            };
        }

        match self.dtype() {
            DType::U8 => dispatch!(U8, u8),
            DType::U32 => dispatch!(U32, u32),
            DType::I16 => dispatch!(I16, i16),
            DType::I32 => dispatch!(I32, i32),
            DType::I64 => dispatch!(I64, i64),
            DType::BF16 => dispatch!(BF16, bf16),
            DType::F16 => dispatch!(F16, f16),
            DType::F32 => dispatch!(F32, f32),
            DType::F64 => dispatch!(F64, f64),
        }
    }

    pub(crate) fn cmp(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: CmpOp,
    ) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (self, rhs) {
                    (Self::$variant(lhs), Self::$variant(rhs)) => Ok(Self::U8(aligned(cmp_map(
                        lhs, lhs_layout, rhs, rhs_layout, op,
                    )?)?)),
                    _ => Err(Error::DTypeMismatch {
                        lhs: self.dtype(),
                        rhs: rhs.dtype(),
                    }),
                }
            };
        }

        match self.dtype() {
            DType::U8 => dispatch!(U8, u8),
            DType::U32 => dispatch!(U32, u32),
            DType::I16 => dispatch!(I16, i16),
            DType::I32 => dispatch!(I32, i32),
            DType::I64 => dispatch!(I64, i64),
            DType::BF16 => dispatch!(BF16, bf16),
            DType::F16 => dispatch!(F16, f16),
            DType::F32 => dispatch!(F32, f32),
            DType::F64 => dispatch!(F64, f64),
        }
    }

    pub(crate) fn cmp_scalar<T: WithDType>(
        &self,
        layout: &Layout,
        scalar: T,
        op: CmpOp,
    ) -> Result<Self> {
        let scalar = T::into_cpu_storage(vec![scalar])?;
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (self, scalar) {
                    (Self::$variant(values), Self::$variant(scalar)) => Ok(Self::U8(aligned(
                        cmp_scalar_map(values, layout, scalar.as_slice()[0], op)?,
                    )?)),
                    _ => Err(Error::DTypeMismatch {
                        lhs: self.dtype(),
                        rhs: T::DTYPE,
                    }),
                }
            };
        }

        match self.dtype() {
            DType::U8 => dispatch!(U8, u8),
            DType::U32 => dispatch!(U32, u32),
            DType::I16 => dispatch!(I16, i16),
            DType::I32 => dispatch!(I32, i32),
            DType::I64 => dispatch!(I64, i64),
            DType::BF16 => dispatch!(BF16, bf16),
            DType::F16 => dispatch!(F16, f16),
            DType::F32 => dispatch!(F32, f32),
            DType::F64 => dispatch!(F64, f64),
        }
    }

    pub(crate) fn where_cond(
        condition: &Self,
        condition_layout: &Layout,
        on_true: &Self,
        true_layout: &Layout,
        on_false: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        let Self::U8(condition) = condition else {
            return Err(Error::UnexpectedDType {
                expected: DType::U8,
                actual: condition.dtype(),
            });
        };

        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                match (on_true, on_false) {
                    (Self::$variant(on_true), Self::$variant(on_false)) => {
                        Ok(Self::$variant(aligned(where_map(
                            condition,
                            condition_layout,
                            on_true,
                            true_layout,
                            on_false,
                            false_layout,
                        )?)?))
                    }
                    _ => Err(Error::DTypeMismatch {
                        lhs: on_true.dtype(),
                        rhs: on_false.dtype(),
                    }),
                }
            };
        }

        match on_true.dtype() {
            DType::U8 => dispatch!(U8, u8),
            DType::U32 => dispatch!(U32, u32),
            DType::I16 => dispatch!(I16, i16),
            DType::I32 => dispatch!(I32, i32),
            DType::I64 => dispatch!(I64, i64),
            DType::BF16 => dispatch!(BF16, bf16),
            DType::F16 => dispatch!(F16, f16),
            DType::F32 => dispatch!(F32, f32),
            DType::F64 => dispatch!(F64, f64),
        }
    }

    pub(crate) fn reduce_dim(
        &self,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        op: ReduceOp,
    ) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    match op {
                        ReduceOp::ArgMin | ReduceOp::ArgMax => Ok(Self::I64(aligned(
                            arg_reduce_map(values, layout, dim, keepdim, op)?,
                        )?)),
                        ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max => Ok(Self::$variant(
                            aligned(reduce_map(values, layout, dim, keepdim, op)?)?,
                        )),
                    }
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn reduce_all(&self, layout: &Layout, op: ReduceOp) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(reduce_all_map(
                        values, layout, op,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn mean_dim(&self, layout: &Layout, dim: usize, keepdim: bool) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(mean_map(
                        values, layout, dim, keepdim,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn mean_all(&self, layout: &Layout) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(mean_all_map(values, layout)?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn var_dim(&self, layout: &Layout, dim: usize, keepdim: bool) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(var_map(
                        values, layout, dim, keepdim,
                    )?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn unary(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(unary_map(values, layout, op)?)?))
                } else {
                    unreachable!()
                }
            };
        }

        match self {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn cat(
        inputs: &[(&Self, &Layout)],
        output_shape: &Shape,
        dim: usize,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::EmptyTensorList);
        }

        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {{
                let mut typed = Vec::with_capacity(inputs.len());
                for (storage, layout) in inputs {
                    match storage {
                        Self::$variant(values) => typed.push((values.as_slice(), *layout)),
                        _ => {
                            return Err(Error::DTypeMismatch {
                                lhs: inputs[0].0.dtype(),
                                rhs: storage.dtype(),
                            });
                        }
                    }
                }
                Ok(Self::$variant(aligned(cat_map(
                    &typed,
                    output_shape.dims(),
                    dim,
                )?)?))
            }};
        }

        match inputs[0].0 {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }

    pub(crate) fn stack_dim0(inputs: &[(&Self, &Layout)], output_shape: &Shape) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::EmptyTensorList);
        }

        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {{
                let mut output = Vec::<$ty>::with_capacity(output_shape.elem_count());
                for (storage, layout) in inputs {
                    let Self::$variant(values) = storage else {
                        return Err(Error::DTypeMismatch {
                            lhs: inputs[0].0.dtype(),
                            rhs: storage.dtype(),
                        });
                    };
                    if layout.elem_count() == 0 {
                        continue;
                    }
                    if let Some((start, end)) = layout.contiguous_offsets() {
                        output.extend_from_slice(
                            values.get(start..end).ok_or(Error::StorageOutOfBounds)?,
                        );
                    } else {
                        for index in layout.strided_index() {
                            output.push(*values.get(index).ok_or(Error::StorageOutOfBounds)?);
                        }
                    }
                }
                Ok(Self::$variant(aligned(output)?))
            }};
        }

        match inputs[0].0 {
            Self::U8(_) => dispatch!(U8, u8),
            Self::U32(_) => dispatch!(U32, u32),
            Self::I16(_) => dispatch!(I16, i16),
            Self::I32(_) => dispatch!(I32, i32),
            Self::I64(_) => dispatch!(I64, i64),
            Self::BF16(_) => dispatch!(BF16, bf16),
            Self::F16(_) => dispatch!(F16, f16),
            Self::F32(_) => dispatch!(F32, f32),
            Self::F64(_) => dispatch!(F64, f64),
        }
    }
}

#[derive(Clone, Copy)]
enum CpuValue {
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
        let count = shape.elem_count();
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
        let count = shape.elem_count();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_backend::buffer::CPU_STORAGE_ALIGNMENT;

    fn assert_aligned(storage: &CpuStorage) {
        match storage {
            CpuStorage::U8(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::U32(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::I16(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::I32(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::I64(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::BF16(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::F16(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::F32(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::F64(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
        }
    }

    #[test]
    fn all_owned_cpu_storage_variants_are_aligned() {
        let device = CpuDevice;
        for dtype in [
            DType::U8,
            DType::U32,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
        ] {
            assert_aligned(&device.zeros(&Shape::from(17), dtype).unwrap());
        }
    }
}
