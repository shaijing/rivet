use super::buffer::AlignedBufferBuilder;
use super::storage::{aligned, CpuStorage};
use super::utils::{
    arg_reduce_map, binary_map, binary_scalar_map, cat_map, cmp_map, cmp_scalar_map, mean_all_map,
    mean_map, reduce_all_map, reduce_map, unary_map, var_map, where_map, ValidatedValues,
};
use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::{DType, Error, Layout, Result, Shape, WithDType};

impl CpuStorage {
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
                let mut builder = AlignedBufferBuilder::new(output_shape.checked_elem_count()?)?;
                cat_map(&typed, output_shape.dims(), dim, |value| {
                    builder.write_next(value)
                })?;
                Ok(Self::$variant(builder.finish()?))
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
                let mut builder = AlignedBufferBuilder::new(output_shape.checked_elem_count()?)?;
                for (storage, layout) in inputs {
                    let Self::$variant(values) = storage else {
                        return Err(Error::DTypeMismatch {
                            lhs: inputs[0].0.dtype(),
                            rhs: storage.dtype(),
                        });
                    };
                    let values = ValidatedValues::new(values.as_slice(), layout)?;
                    if layout.checked_elem_count()? == 0 {
                        continue;
                    }
                    if let Some((start, end)) = layout.contiguous_offsets() {
                        builder.extend_from_slice(
                            values
                                .as_slice()
                                .get(start..end)
                                .ok_or(Error::StorageOutOfBounds)?,
                        )?;
                    } else {
                        for index in layout.strided_index() {
                            builder.write_next(values.read(index))?;
                        }
                    }
                }
                Ok(Self::$variant(builder.finish()?))
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
