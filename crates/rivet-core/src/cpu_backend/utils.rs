use super::buffer::{AlignedBuffer, AlignedBufferBuilder};
use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::storage::validate_layout_for_storage;
use crate::{DType, Error, Layout, Result};

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

/// A typed view over storage whose layout has been checked once.
///
/// The backend iterators only produce offsets addressed by the validated
/// layout. Keeping the unchecked read here prevents bounds-check branches from
/// being repeated in every element of strided kernels.
struct ValidatedValues<'a, T> {
    values: &'a [T],
}

impl<'a, T: Copy> ValidatedValues<'a, T> {
    fn new(values: &'a [T], layout: &Layout) -> Result<Self> {
        validate_layout_for_storage(layout, values.len())?;
        Ok(Self { values })
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        self.values
    }

    #[inline]
    fn read(&self, index: usize) -> T {
        debug_assert!(index < self.values.len());
        // SAFETY: construction validates that every offset addressable by the
        // corresponding layout is within `values`. Callers pass offsets from
        // that layout's strided iterator or coordinate calculations.
        unsafe { *self.values.get_unchecked(index) }
    }
}

pub trait BinaryElement: Copy + Send + Sync + 'static {
    fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self>;
}

pub trait CmpElement: Copy + Send + Sync + 'static {
    fn compare(op: CmpOp, lhs: Self, rhs: Self) -> u8;
}

pub trait ReduceElement: Copy + Send + Sync + 'static {
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn less(self, rhs: Self) -> bool;
    fn greater(self, rhs: Self) -> bool;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
    fn is_nan(self) -> bool {
        false
    }
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
                    BinaryOp::Minimum => lhs.min(rhs),
                    BinaryOp::Maximum => lhs.max(rhs),
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
                    BinaryOp::Minimum => lhs.min(rhs),
                    BinaryOp::Maximum => lhs.max(rhs),
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
                    BinaryOp::Minimum => lhs.min(rhs),
                    BinaryOp::Maximum => lhs.max(rhs),
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

macro_rules! impl_cmp_element {
    ($ty:ty) => {
        impl CmpElement for $ty {
            fn compare(op: CmpOp, lhs: Self, rhs: Self) -> u8 {
                match op {
                    CmpOp::Eq => u8::from(lhs == rhs),
                    CmpOp::Ne => u8::from(lhs != rhs),
                    CmpOp::Lt => u8::from(lhs < rhs),
                    CmpOp::Le => u8::from(lhs <= rhs),
                    CmpOp::Gt => u8::from(lhs > rhs),
                    CmpOp::Ge => u8::from(lhs >= rhs),
                }
            }
        }
    };
}

macro_rules! impl_reduce_int {
    ($ty:ty) => {
        impl ReduceElement for $ty {
            fn zero() -> Self {
                0
            }

            fn add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            fn less(self, rhs: Self) -> bool {
                self < rhs
            }

            fn greater(self, rhs: Self) -> bool {
                self > rhs
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    };
}

macro_rules! impl_reduce_float {
    ($ty:ty) => {
        impl ReduceElement for $ty {
            fn zero() -> Self {
                0.0
            }

            fn add(self, rhs: Self) -> Self {
                self + rhs
            }

            fn less(self, rhs: Self) -> bool {
                self < rhs
            }

            fn greater(self, rhs: Self) -> bool {
                self > rhs
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn from_f64(value: f64) -> Self {
                value as $ty
            }

            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }
    };
}

impl_cmp_element!(u8);
impl_cmp_element!(u32);
impl_cmp_element!(i16);
impl_cmp_element!(i32);
impl_cmp_element!(i64);
impl_cmp_element!(f32);
impl_cmp_element!(f64);
impl_reduce_int!(u8);
impl_reduce_int!(u32);
impl_reduce_int!(i16);
impl_reduce_int!(i32);
impl_reduce_int!(i64);
impl_reduce_float!(f32);
impl_reduce_float!(f64);

impl BinaryElement for half::f16 {
    fn apply(op: BinaryOp, lhs: Self, rhs: Self) -> Result<Self> {
        Ok(match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
            BinaryOp::Minimum => lhs.min(rhs),
            BinaryOp::Maximum => lhs.max(rhs),
        })
    }
}

impl CmpElement for half::f16 {
    fn compare(op: CmpOp, lhs: Self, rhs: Self) -> u8 {
        match op {
            CmpOp::Eq => u8::from(lhs == rhs),
            CmpOp::Ne => u8::from(lhs != rhs),
            CmpOp::Lt => u8::from(lhs < rhs),
            CmpOp::Le => u8::from(lhs <= rhs),
            CmpOp::Gt => u8::from(lhs > rhs),
            CmpOp::Ge => u8::from(lhs >= rhs),
        }
    }
}

impl ReduceElement for half::f16 {
    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn less(self, rhs: Self) -> bool {
        self < rhs
    }

    fn greater(self, rhs: Self) -> bool {
        self > rhs
    }

    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }

    fn is_nan(self) -> bool {
        self.is_nan()
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
            BinaryOp::Minimum => lhs.min(rhs),
            BinaryOp::Maximum => lhs.max(rhs),
        })
    }
}

impl CmpElement for half::bf16 {
    fn compare(op: CmpOp, lhs: Self, rhs: Self) -> u8 {
        match op {
            CmpOp::Eq => u8::from(lhs == rhs),
            CmpOp::Ne => u8::from(lhs != rhs),
            CmpOp::Lt => u8::from(lhs < rhs),
            CmpOp::Le => u8::from(lhs <= rhs),
            CmpOp::Gt => u8::from(lhs > rhs),
            CmpOp::Ge => u8::from(lhs >= rhs),
        }
    }
}

impl ReduceElement for half::bf16 {
    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn less(self, rhs: Self) -> bool {
        self < rhs
    }

    fn greater(self, rhs: Self) -> bool {
        self > rhs
    }

    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }

    fn is_nan(self) -> bool {
        self.is_nan()
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

pub(crate) fn binary_map<T: BinaryElement>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
    op: BinaryOp,
) -> Result<AlignedBuffer<T>> {
    if lhs_layout.shape() != rhs_layout.shape() {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    let lhs_values = ValidatedValues::new(lhs, lhs_layout)?;
    let rhs_values = ValidatedValues::new(rhs, rhs_layout)?;

    if let (Some((lhs_start, lhs_end)), Some((rhs_start, rhs_end))) = (
        lhs_layout.contiguous_offsets(),
        rhs_layout.contiguous_offsets(),
    ) {
        let lhs = checked_slice(lhs_values.as_slice(), lhs_start, lhs_end)?;
        let rhs = checked_slice(rhs_values.as_slice(), rhs_start, rhs_end)?;
        if lhs.len() != rhs.len() {
            return Err(Error::ShapeMismatchBinary {
                lhs: vec![lhs.len()],
                rhs: vec![rhs.len()],
            });
        }
        return aligned_try_iter(
            lhs.iter()
                .zip(rhs)
                .map(|(&lhs, &rhs)| T::apply(op, lhs, rhs)),
        );
    }

    aligned_try_iter(
        lhs_layout
            .strided_index()
            .zip(rhs_layout.strided_index())
            .map(|(lhs_index, rhs_index)| {
                let lhs = lhs_values.read(lhs_index);
                let rhs = rhs_values.read(rhs_index);
                T::apply(op, lhs, rhs)
            }),
    )
}

pub(crate) fn binary_scalar_map<T: BinaryElement>(
    values: &[T],
    layout: &Layout,
    scalar: T,
    op: BinaryOp,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    if let Some((start, end)) = layout.contiguous_offsets() {
        return aligned_try_iter(
            checked_slice(values.as_slice(), start, end)?
                .iter()
                .map(|&value| T::apply(op, value, scalar)),
        );
    }

    aligned_try_iter(layout.strided_index().map(|index| {
        let value = values.read(index);
        T::apply(op, value, scalar)
    }))
}

pub(crate) fn unary_map<T: UnaryElement>(
    values: &[T],
    layout: &Layout,
    op: UnaryOp,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    if let Some((start, end)) = layout.contiguous_offsets() {
        return checked_slice(values.as_slice(), start, end).and_then(|values| {
            aligned_try_iter(values.iter().map(|&value| Ok(T::apply(op, value))))
        });
    }

    aligned_try_iter(
        layout
            .strided_index()
            .map(|index| Ok(T::apply(op, values.read(index)))),
    )
}

pub(crate) fn cmp_map<T: CmpElement>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
    op: CmpOp,
) -> Result<AlignedBuffer<u8>> {
    if lhs_layout.shape() != rhs_layout.shape() {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    let lhs_values = ValidatedValues::new(lhs, lhs_layout)?;
    let rhs_values = ValidatedValues::new(rhs, rhs_layout)?;

    aligned_try_iter(
        lhs_layout
            .strided_index()
            .zip(rhs_layout.strided_index())
            .map(|(lhs_index, rhs_index)| {
                let lhs = lhs_values.read(lhs_index);
                let rhs = rhs_values.read(rhs_index);
                Ok(T::compare(op, lhs, rhs))
            }),
    )
}

pub(crate) fn cmp_scalar_map<T: CmpElement>(
    values: &[T],
    layout: &Layout,
    scalar: T,
    op: CmpOp,
) -> Result<AlignedBuffer<u8>> {
    let values = ValidatedValues::new(values, layout)?;
    aligned_try_iter(layout.strided_index().map(|index| {
        let value = values.read(index);
        Ok(T::compare(op, value, scalar))
    }))
}

pub(crate) fn where_map<T: Copy>(
    condition: &[u8],
    condition_layout: &Layout,
    on_true: &[T],
    true_layout: &Layout,
    on_false: &[T],
    false_layout: &Layout,
) -> Result<AlignedBuffer<T>> {
    if condition_layout.shape() != true_layout.shape()
        || condition_layout.shape() != false_layout.shape()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: condition_layout.dims().to_vec(),
            rhs: true_layout.dims().to_vec(),
        });
    }
    let condition = ValidatedValues::new(condition, condition_layout)?;
    let on_true = ValidatedValues::new(on_true, true_layout)?;
    let on_false = ValidatedValues::new(on_false, false_layout)?;

    aligned_try_iter(
        condition_layout
            .strided_index()
            .zip(true_layout.strided_index())
            .zip(false_layout.strided_index())
            .map(|((condition_index, true_index), false_index)| {
                let condition = condition.read(condition_index);
                let value = if condition != 0 {
                    on_true.read(true_index)
                } else {
                    on_false.read(false_index)
                };
                Ok(value)
            }),
    )
}

fn output_coordinates(dims: &[usize], mut index: usize) -> Vec<usize> {
    let mut coordinates = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        coordinates[axis] = index % dims[axis];
        index /= dims[axis];
    }
    coordinates
}

fn output_dims_for_dim(dims: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    if keepdim {
        let mut output = dims.to_vec();
        output[dim] = 1;
        output
    } else {
        dims.iter()
            .enumerate()
            .filter_map(|(axis, &size)| (axis != dim).then_some(size))
            .collect()
    }
}

fn input_base_index(
    layout: &Layout,
    dim: usize,
    output_coordinates: &[usize],
    keepdim: bool,
) -> usize {
    let mut output_axis = 0;
    let mut index = layout.start_offset();
    for axis in 0..layout.dims().len() {
        if axis == dim {
            continue;
        }
        let coordinate = if keepdim {
            output_coordinates[axis]
        } else {
            let coordinate = output_coordinates[output_axis];
            output_axis += 1;
            coordinate
        };
        index += coordinate * layout.stride()[axis];
    }
    index
}

fn select_extreme<T: ReduceElement>(current: T, candidate: T, use_min: bool) -> T {
    // NaN is sticky and the first NaN wins. This keeps min/max and argmin/argmax
    // deterministic while making accidental NaNs visible to callers.
    if current.is_nan() {
        return current;
    }
    if candidate.is_nan() {
        return candidate;
    }
    let better = if use_min {
        candidate.less(current)
    } else {
        candidate.greater(current)
    };
    better.then_some(candidate).unwrap_or(current)
}

pub(crate) fn reduce_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
    op: ReduceOp,
) -> Result<AlignedBuffer<T>> {
    if dim >= layout.dims().len() {
        return Err(Error::InvalidDim {
            dim,
            rank: layout.dims().len(),
        });
    }
    let output_dims = output_dims_for_dim(layout.dims(), dim, keepdim);
    let output_len = output_dims.iter().product::<usize>();
    let reduce_len = layout.dims()[dim];
    if reduce_len == 0 && matches!(op, ReduceOp::Min | ReduceOp::Max) {
        return Err(Error::EmptyReduction {
            op: if op == ReduceOp::Min { "min" } else { "max" },
            dim,
        });
    }
    let values = ValidatedValues::new(values, layout)?;
    let mut builder = AlignedBufferBuilder::new(output_len)?;
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates, keepdim);
        match op {
            ReduceOp::Sum => {
                let mut value = T::zero();
                for offset in 0..reduce_len {
                    let index = base + offset * layout.stride()[dim];
                    value = value.add(values.read(index));
                }
                builder.write_next(value)?;
            }
            ReduceOp::Min | ReduceOp::Max => {
                let use_min = op == ReduceOp::Min;
                let first = values.read(base);
                let mut value = first;
                for offset in 1..reduce_len {
                    let index = base + offset * layout.stride()[dim];
                    value = select_extreme(value, values.read(index), use_min);
                }
                builder.write_next(value)?;
            }
            ReduceOp::ArgMin | ReduceOp::ArgMax => {
                unreachable!("arg reductions use arg_reduce_map")
            }
        }
    }
    builder.finish()
}

pub(crate) fn arg_reduce_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
    op: ReduceOp,
) -> Result<AlignedBuffer<i64>> {
    if dim >= layout.dims().len() {
        return Err(Error::InvalidDim {
            dim,
            rank: layout.dims().len(),
        });
    }
    let output_dims = output_dims_for_dim(layout.dims(), dim, keepdim);
    let output_len = output_dims.iter().product::<usize>();
    let reduce_len = layout.dims()[dim];
    if reduce_len == 0 {
        return Err(Error::EmptyReduction {
            op: if op == ReduceOp::ArgMin {
                "argmin"
            } else {
                "argmax"
            },
            dim,
        });
    }
    let values = ValidatedValues::new(values, layout)?;
    let mut builder = AlignedBufferBuilder::new(output_len)?;
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates, keepdim);
        let use_min = op == ReduceOp::ArgMin;
        let mut best_index = 0usize;
        let mut best = values.read(base);
        for offset in 1..reduce_len {
            let index = base + offset * layout.stride()[dim];
            let candidate = values.read(index);
            let candidate_is_better = if best.is_nan() {
                false
            } else if candidate.is_nan() {
                true
            } else if use_min {
                candidate.less(best)
            } else {
                candidate.greater(best)
            };
            if candidate_is_better {
                let selected = select_extreme(best, candidate, use_min);
                best = selected;
                best_index = offset;
            }
        }
        builder.write_next(best_index as i64)?;
    }
    builder.finish()
}

pub(crate) fn reduce_all_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    op: ReduceOp,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    match op {
        ReduceOp::Sum => {
            let mut value = T::zero();
            for index in layout.strided_index() {
                value = value.add(values.read(index));
            }
            aligned_one(value)
        }
        ReduceOp::Min | ReduceOp::Max => {
            let mut indices = layout.strided_index();
            let first_index = indices.next().ok_or(Error::EmptyReduction {
                op: if op == ReduceOp::Min {
                    "min_all"
                } else {
                    "max_all"
                },
                dim: 0,
            })?;
            let mut value = values.read(first_index);
            let use_min = op == ReduceOp::Min;
            for index in indices {
                value = select_extreme(value, values.read(index), use_min);
            }
            aligned_one(value)
        }
        ReduceOp::ArgMin | ReduceOp::ArgMax => {
            unreachable!("arg reductions are dimension-only")
        }
    }
}

pub(crate) fn mean_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
) -> Result<AlignedBuffer<T>> {
    reduce_f64_map(values, layout, dim, keepdim, false)
}

pub(crate) fn var_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
) -> Result<AlignedBuffer<T>> {
    reduce_f64_map(values, layout, dim, keepdim, true)
}

fn reduce_f64_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
    variance: bool,
) -> Result<AlignedBuffer<T>> {
    if dim >= layout.dims().len() {
        return Err(Error::InvalidDim {
            dim,
            rank: layout.dims().len(),
        });
    }
    let output_dims = output_dims_for_dim(layout.dims(), dim, keepdim);
    let output_len = output_dims.iter().product::<usize>();
    let reduce_len = layout.dims()[dim];
    if reduce_len == 0 {
        return Err(Error::EmptyReduction {
            op: if variance { "var" } else { "mean" },
            dim,
        });
    }
    if variance && reduce_len < 2 {
        return Err(Error::InvalidReduction {
            op: "var",
            dim,
            minimum: 2,
            actual: reduce_len,
        });
    }
    let values = ValidatedValues::new(values, layout)?;
    let mut builder = AlignedBufferBuilder::new(output_len)?;
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates, keepdim);
        let mut sum = 0.0;
        for offset in 0..reduce_len {
            let index = base + offset * layout.stride()[dim];
            sum += values.read(index).to_f64();
        }
        let mean = sum / reduce_len as f64;
        let value = if variance {
            let mut squared_error = 0.0;
            for offset in 0..reduce_len {
                let index = base + offset * layout.stride()[dim];
                let delta = values.read(index).to_f64() - mean;
                squared_error += delta * delta;
            }
            squared_error / (reduce_len - 1) as f64
        } else {
            mean
        };
        builder.write_next(T::from_f64(value))?;
    }
    builder.finish()
}

pub(crate) fn mean_all_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    let mut count = 0usize;
    let mut sum = 0.0;
    for index in layout.strided_index() {
        sum += values.read(index).to_f64();
        count += 1;
    }
    if count == 0 {
        return Err(Error::EmptyReduction {
            op: "mean_all",
            dim: 0,
        });
    }
    aligned_one(T::from_f64(sum / count as f64))
}

pub(crate) fn copy_logical<T: Copy>(values: &[T], layout: &Layout) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    if let Some((start, end)) = layout.contiguous_offsets() {
        return checked_slice(values.as_slice(), start, end).and_then(AlignedBuffer::from_slice);
    }

    aligned_try_iter(layout.strided_index().map(|index| Ok(values.read(index))))
}

pub(crate) fn cat_map<T: Copy>(
    inputs: &[(&[T], &Layout)],
    output_shape: &[usize],
    dim: usize,
    mut write: impl FnMut(T) -> Result<()>,
) -> Result<()> {
    let rank = output_shape.len();
    if dim >= rank {
        return Err(Error::InvalidConcatDim { dim, rank });
    }
    let inputs = inputs
        .iter()
        .map(|(values, layout)| Ok((ValidatedValues::new(values, layout)?, *layout)))
        .collect::<Result<Vec<_>>>()?;
    let inner = output_shape[dim + 1..].iter().product::<usize>();
    let outer = output_shape[..dim].iter().product::<usize>();
    let mut logical_indices = inputs
        .iter()
        .map(|(_, layout)| layout.strided_index())
        .collect::<Vec<_>>();

    for _ in 0..outer {
        for ((values, layout), logical_index) in inputs.iter().zip(&mut logical_indices) {
            let input_block = layout.dims()[dim] * inner;
            for _ in 0..input_block {
                let index = logical_index.next().ok_or(Error::StorageOutOfBounds)?;
                write(values.read(index))?;
            }
        }
    }
    Ok(())
}
