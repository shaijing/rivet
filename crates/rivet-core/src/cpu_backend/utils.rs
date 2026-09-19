use crate::ops::{BinaryOp, UnaryOp};
use crate::{DType, Error, Layout, Result};

pub trait BinaryElement: Copy + Send + Sync + 'static {
    fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self>;
}

pub trait UnaryElement: Copy + Send + Sync + 'static {
    fn apply(op: UnaryOp, value: Self) -> Self;
}

macro_rules! impl_unsigned_element {
    ($ty:ty, $dtype:ident) => {
        impl BinaryElement for $ty {
            fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self> {
                Ok(match op {
                    BinaryOp::Add => lhs.wrapping_add(rhs),
                    BinaryOp::Sub => lhs.wrapping_sub(rhs),
                    BinaryOp::Mul => lhs.wrapping_mul(rhs),
                    BinaryOp::Div => {
                        if rhs == 0 {
                            return Err(Error::DivisionByZero {
                                dtype: DType::$dtype,
                            });
                        }
                        lhs.wrapping_div(rhs)
                    }
                })
            }
        }

        impl UnaryElement for $ty {
            fn apply(op: UnaryOp, value: Self) -> Self {
                match op {
                    UnaryOp::Neg => value.wrapping_neg(),
                    UnaryOp::Abs => value,
                }
            }
        }
    };
}

macro_rules! impl_signed_element {
    ($ty:ty, $dtype:ident) => {
        impl BinaryElement for $ty {
            fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self> {
                Ok(match op {
                    BinaryOp::Add => lhs.wrapping_add(rhs),
                    BinaryOp::Sub => lhs.wrapping_sub(rhs),
                    BinaryOp::Mul => lhs.wrapping_mul(rhs),
                    BinaryOp::Div => {
                        if rhs == 0 {
                            return Err(Error::DivisionByZero {
                                dtype: DType::$dtype,
                            });
                        }
                        lhs.wrapping_div(rhs)
                    }
                })
            }
        }

        impl UnaryElement for $ty {
            fn apply(op: UnaryOp, value: Self) -> Self {
                match op {
                    UnaryOp::Neg => value.wrapping_neg(),
                    UnaryOp::Abs => value.wrapping_abs(),
                }
            }
        }
    };
}

macro_rules! impl_float_element {
    ($ty:ty, $dtype:ident) => {
        impl BinaryElement for $ty {
            fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self> {
                Ok(match op {
                    BinaryOp::Add => lhs + rhs,
                    BinaryOp::Sub => lhs - rhs,
                    BinaryOp::Mul => lhs * rhs,
                    BinaryOp::Div => lhs / rhs,
                })
            }
        }

        impl UnaryElement for $ty {
            fn apply(op: UnaryOp, value: Self) -> Self {
                match op {
                    UnaryOp::Neg => -value,
                    UnaryOp::Abs => value.abs(),
                }
            }
        }
    };
}

impl_unsigned_element!(u8, U8);
impl_unsigned_element!(u32, U32);
impl_signed_element!(i16, I16);
impl_signed_element!(i32, I32);
impl_signed_element!(i64, I64);
impl_float_element!(f32, F32);
impl_float_element!(f64, F64);

impl BinaryElement for half::f16 {
    fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self> {
        Ok(match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        })
    }
}

impl UnaryElement for half::f16 {
    fn apply(op: UnaryOp, value: Self) -> Self {
        match op {
            UnaryOp::Neg => -value,
            UnaryOp::Abs => half::f16::from_f32(value.to_f32().abs()),
        }
    }
}

impl BinaryElement for half::bf16 {
    fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self> {
        Ok(match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        })
    }
}

impl UnaryElement for half::bf16 {
    fn apply(op: UnaryOp, value: Self) -> Self {
        match op {
            UnaryOp::Neg => -value,
            UnaryOp::Abs => half::bf16::from_f32(value.to_f32().abs()),
        }
    }
}

fn checked_slice<'a, T>(data: &'a [T], start: usize, end: usize) -> Result<&'a [T]> {
    data.get(start..end).ok_or(Error::StorageOutOfBounds)
}

pub fn binary_map<T: BinaryElement>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
    op: BinaryOp,
) -> Result<Vec<T>> {
    if lhs_layout.shape() != rhs_layout.shape() {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }

    if let (Some((lhs_start, lhs_end)), Some((rhs_start, rhs_end))) = (
        lhs_layout.contiguous_offsets(),
        rhs_layout.contiguous_offsets(),
    ) {
        let lhs = checked_slice(lhs, lhs_start, lhs_end)?;
        let rhs = checked_slice(rhs, rhs_start, rhs_end)?;
        if lhs.len() != rhs.len() {
            return Err(Error::ShapeMismatchBinary {
                lhs: vec![lhs.len()],
                rhs: vec![rhs.len()],
            });
        }
        return lhs
            .iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| T::apply(op, lhs, rhs))
            .collect();
    }

    let mut output = Vec::with_capacity(lhs_layout.elem_count());
    for (lhs_index, rhs_index) in lhs_layout.strided_index().zip(rhs_layout.strided_index()) {
        let lhs = *lhs.get(lhs_index).ok_or(Error::StorageOutOfBounds)?;
        let rhs = *rhs.get(rhs_index).ok_or(Error::StorageOutOfBounds)?;
        output.push(T::apply(op, lhs, rhs)?);
    }
    Ok(output)
}

pub fn binary_scalar_map<T: BinaryElement>(
    values: &[T],
    layout: &Layout,
    scalar: T,
    op: BinaryOp,
) -> Result<Vec<T>> {
    if let Some((start, end)) = layout.contiguous_offsets() {
        return checked_slice(values, start, end)?
            .iter()
            .map(|&value| T::apply(op, value, scalar))
            .collect();
    }

    let mut output = Vec::with_capacity(layout.elem_count());
    for index in layout.strided_index() {
        let value = *values.get(index).ok_or(Error::StorageOutOfBounds)?;
        output.push(T::apply(op, value, scalar)?);
    }
    Ok(output)
}

pub fn unary_map<T: UnaryElement>(values: &[T], layout: &Layout, op: UnaryOp) -> Result<Vec<T>> {
    if let Some((start, end)) = layout.contiguous_offsets() {
        return checked_slice(values, start, end)
            .map(|values| values.iter().map(|&value| T::apply(op, value)).collect());
    }

    let mut output = Vec::with_capacity(layout.elem_count());
    for index in layout.strided_index() {
        let value = *values.get(index).ok_or(Error::StorageOutOfBounds)?;
        output.push(T::apply(op, value));
    }
    Ok(output)
}

pub fn copy_logical<T: Copy>(values: &[T], layout: &Layout) -> Result<Vec<T>> {
    if let Some((start, end)) = layout.contiguous_offsets() {
        return checked_slice(values, start, end).map(ToOwned::to_owned);
    }

    layout
        .strided_index()
        .map(|index| values.get(index).copied().ok_or(Error::StorageOutOfBounds))
        .collect()
}

pub fn cat_map<T: Copy>(
    inputs: &[(&[T], &Layout)],
    output_shape: &[usize],
    dim: usize,
) -> Result<Vec<T>> {
    let rank = output_shape.len();
    if dim >= rank {
        return Err(Error::InvalidConcatDim { dim, rank });
    }
    let inner = output_shape[dim + 1..].iter().product::<usize>();
    let outer = output_shape[..dim].iter().product::<usize>();
    let mut output = Vec::with_capacity(output_shape.iter().product());
    let mut logical_indices = inputs
        .iter()
        .map(|(_, layout)| layout.strided_index())
        .collect::<Vec<_>>();

    for _ in 0..outer {
        for ((values, layout), logical_index) in inputs.iter().zip(&mut logical_indices) {
            let input_block = layout.dims()[dim] * inner;
            for _ in 0..input_block {
                let index = logical_index.next().ok_or(Error::StorageOutOfBounds)?;
                output.push(*values.get(index).ok_or(Error::StorageOutOfBounds)?);
            }
        }
    }
    Ok(output)
}
