use super::buffer::{AlignedBuffer, AlignedBufferBuilder};
use super::utils::{ReduceElement, ValidatedValues};
use crate::{Error, Layout, Result, Shape};

use super::storage::{aligned, CpuStorage};
use half::{bf16, f16};

fn aligned_try_iter<T, I>(iter: I) -> Result<AlignedBuffer<T>>
where
    I: ExactSizeIterator<Item = Result<T>>,
{
    let mut builder = AlignedBufferBuilder::new(iter.len())?;
    for value in iter {
        builder.write_next(value?)?;
    }
    builder.finish()
}

fn aligned_one<T>(value: T) -> Result<AlignedBuffer<T>> {
    let mut builder = AlignedBufferBuilder::new(1)?;
    builder.write_next(value)?;
    builder.finish()
}

/// Floating-point element contract for model-oriented math kernels.
///
/// The backend computes transcendental operations in `f64` and casts back to
/// the storage dtype. This gives all supported floating dtypes one explicit
/// implementation while preserving the tensor's dtype at the API boundary.
pub trait FloatElement: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

macro_rules! impl_float_element {
    ($ty:ty) => {
        impl FloatElement for $ty {
            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    };
}

impl_float_element!(f32);
impl_float_element!(f64);

impl FloatElement for half::f16 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl FloatElement for half::bf16 {
    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }
}

fn logical_index(layout: &Layout, coordinates: &[usize]) -> Result<usize> {
    let mut index = layout.start_offset();
    for (&coordinate, &stride) in coordinates.iter().zip(layout.stride()) {
        index = index
            .checked_add(
                coordinate
                    .checked_mul(stride)
                    .ok_or(Error::StorageOutOfBounds)?,
            )
            .ok_or(Error::StorageOutOfBounds)?;
    }
    Ok(index)
}

fn output_coordinates(dims: &[usize], mut index: usize) -> Vec<usize> {
    let mut coordinates = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        coordinates[axis] = index % dims[axis];
        index /= dims[axis];
    }
    coordinates
}

fn output_dims_for_dim(dims: &[usize], dim: usize) -> Vec<usize> {
    dims.iter()
        .enumerate()
        .filter_map(|(axis, &size)| (axis != dim).then_some(size))
        .collect()
}

fn input_base_index(layout: &Layout, dim: usize, output_coordinates: &[usize]) -> Result<usize> {
    let mut input_coordinates = vec![0; layout.dims().len()];
    let mut output_axis = 0;
    for (axis, coordinate) in input_coordinates.iter_mut().enumerate() {
        if axis != dim {
            *coordinate = output_coordinates[output_axis];
            output_axis += 1;
        }
    }
    logical_index(layout, &input_coordinates)
}

impl CpuStorage {
    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty, $convert:expr) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(affine_map(
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
                    Ok(Self::$variant(aligned(elu_map(values, layout, alpha)?)?))
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
                    Ok(Self::$variant(aligned(powf_map(
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
                        pow_map(lhs, lhs_layout, rhs, rhs_layout)?,
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
                        dot_map(lhs, lhs_layout, rhs, rhs_layout)?,
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
                    Ok(Self::$variant(aligned(norm_map(values, layout)?)?))
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
                    Ok(Self::$variant(aligned(cumsum_map(values, layout, dim)?)?))
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
                    Ok(Self::$variant(aligned(log_sum_exp_map(
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
}

pub(crate) fn affine_map<T: Copy>(
    values: &[T],
    layout: &Layout,
    map: impl Fn(T) -> T,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    aligned_try_iter(
        layout
            .strided_index()
            .map(|index| Ok(map(values.read(index)))),
    )
}

pub(crate) fn elu_map<T: FloatElement>(
    values: &[T],
    layout: &Layout,
    alpha: f64,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    aligned_try_iter(layout.strided_index().map(|index| {
        let value = values.read(index).to_f64();
        let output = if value > 0.0 {
            value
        } else {
            alpha * value.exp_m1()
        };
        Ok(T::from_f64(output))
    }))
}

pub(crate) fn powf_map<T: FloatElement>(
    values: &[T],
    layout: &Layout,
    exponent: f64,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    aligned_try_iter(
        layout
            .strided_index()
            .map(|index| Ok(T::from_f64(values.read(index).to_f64().powf(exponent)))),
    )
}

pub fn pow_map<T: FloatElement>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
) -> Result<AlignedBuffer<T>> {
    if lhs_layout.shape() != rhs_layout.shape() {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    let lhs = ValidatedValues::new(lhs, lhs_layout)?;
    let rhs = ValidatedValues::new(rhs, rhs_layout)?;
    aligned_try_iter(
        lhs_layout
            .strided_index()
            .zip(rhs_layout.strided_index())
            .map(|(lhs_index, rhs_index)| {
                let base = lhs.read(lhs_index).to_f64();
                let exponent = rhs.read(rhs_index).to_f64();
                Ok(T::from_f64(base.powf(exponent)))
            }),
    )
}

pub fn dot_map<T: FloatElement>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
) -> Result<AlignedBuffer<T>> {
    if lhs_layout.dims().len() != 1
        || rhs_layout.dims().len() != 1
        || lhs_layout.shape() != rhs_layout.shape()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    let lhs = ValidatedValues::new(lhs, lhs_layout)?;
    let rhs = ValidatedValues::new(rhs, rhs_layout)?;
    let mut sum = 0.0;
    for (lhs_index, rhs_index) in lhs_layout.strided_index().zip(rhs_layout.strided_index()) {
        sum += lhs.read(lhs_index).to_f64() * rhs.read(rhs_index).to_f64();
    }
    aligned_one(T::from_f64(sum))
}

pub(crate) fn norm_map<T: FloatElement>(values: &[T], layout: &Layout) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    let mut sum = 0.0;
    for index in layout.strided_index() {
        let value = values.read(index).to_f64();
        sum += value * value;
    }
    aligned_one(T::from_f64(sum.sqrt()))
}

pub(crate) fn cumsum_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
) -> Result<AlignedBuffer<T>> {
    if dim >= layout.dims().len() {
        return Err(Error::InvalidDim {
            dim,
            rank: layout.dims().len(),
        });
    }

    let values = ValidatedValues::new(values, layout)?;
    let output_len = layout.checked_elem_count()?;
    let mut builder = AlignedBufferBuilder::new(output_len)?;
    for output_index in 0..output_len {
        let coordinates = output_coordinates(layout.dims(), output_index);
        let mut sum = T::zero();
        for end in 0..=coordinates[dim] {
            let mut input_coordinates = coordinates.clone();
            input_coordinates[dim] = end;
            let index = logical_index(layout, &input_coordinates)?;
            sum = sum.add(values.read(index));
        }
        builder.write_next(sum)?;
    }
    builder.finish()
}

pub fn log_sum_exp_map<T: FloatElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
) -> Result<AlignedBuffer<T>> {
    if dim >= layout.dims().len() {
        return Err(Error::InvalidDim {
            dim,
            rank: layout.dims().len(),
        });
    }
    let reduce_len = layout.dims()[dim];
    if reduce_len == 0 {
        return Err(Error::EmptyReduction {
            op: "log_sum_exp",
            dim,
        });
    }

    let values = ValidatedValues::new(values, layout)?;
    let output_dims = output_dims_for_dim(layout.dims(), dim);
    let output_len = Shape::from(output_dims.clone()).checked_elem_count()?;
    let mut builder = AlignedBufferBuilder::new(output_len)?;
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates)?;
        let mut maximum = f64::NEG_INFINITY;
        let mut has_nan = false;
        for offset in 0..reduce_len {
            let index = base
                .checked_add(
                    offset
                        .checked_mul(layout.stride()[dim])
                        .ok_or(Error::StorageOutOfBounds)?,
                )
                .ok_or(Error::StorageOutOfBounds)?;
            let value = values.read(index).to_f64();
            has_nan |= value.is_nan();
            maximum = maximum.max(value);
        }

        let result = if has_nan {
            f64::NAN
        } else if maximum == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            let mut sum = 0.0;
            for offset in 0..reduce_len {
                let index = base
                    .checked_add(
                        offset
                            .checked_mul(layout.stride()[dim])
                            .ok_or(Error::StorageOutOfBounds)?,
                    )
                    .ok_or(Error::StorageOutOfBounds)?;
                sum += (values.read(index).to_f64() - maximum).exp();
            }
            maximum + sum.ln()
        };
        builder.write_next(T::from_f64(result))?;
    }
    builder.finish()
}
