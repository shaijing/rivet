use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::{DType, Error, Layout, Result};

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

pub fn cmp_map<T: CmpElement>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
    op: CmpOp,
) -> Result<Vec<u8>> {
    if lhs_layout.shape() != rhs_layout.shape() {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }

    let mut output = Vec::with_capacity(lhs_layout.elem_count());
    for (lhs_index, rhs_index) in lhs_layout.strided_index().zip(rhs_layout.strided_index()) {
        let lhs = *lhs.get(lhs_index).ok_or(Error::StorageOutOfBounds)?;
        let rhs = *rhs.get(rhs_index).ok_or(Error::StorageOutOfBounds)?;
        output.push(T::compare(op, lhs, rhs));
    }
    Ok(output)
}

pub fn cmp_scalar_map<T: CmpElement>(
    values: &[T],
    layout: &Layout,
    scalar: T,
    op: CmpOp,
) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(layout.elem_count());
    for index in layout.strided_index() {
        let value = *values.get(index).ok_or(Error::StorageOutOfBounds)?;
        output.push(T::compare(op, value, scalar));
    }
    Ok(output)
}

pub fn where_map<T: Copy>(
    condition: &[u8],
    condition_layout: &Layout,
    on_true: &[T],
    true_layout: &Layout,
    on_false: &[T],
    false_layout: &Layout,
) -> Result<Vec<T>> {
    if condition_layout.shape() != true_layout.shape()
        || condition_layout.shape() != false_layout.shape()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: condition_layout.dims().to_vec(),
            rhs: true_layout.dims().to_vec(),
        });
    }

    let mut output = Vec::with_capacity(condition_layout.elem_count());
    for ((condition_index, true_index), false_index) in condition_layout
        .strided_index()
        .zip(true_layout.strided_index())
        .zip(false_layout.strided_index())
    {
        let condition = *condition
            .get(condition_index)
            .ok_or(Error::StorageOutOfBounds)?;
        let value = if condition != 0 {
            *on_true.get(true_index).ok_or(Error::StorageOutOfBounds)?
        } else {
            *on_false.get(false_index).ok_or(Error::StorageOutOfBounds)?
        };
        output.push(value);
    }
    Ok(output)
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

pub fn reduce_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
    op: ReduceOp,
) -> Result<Vec<T>> {
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
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates, keepdim);
        match op {
            ReduceOp::Sum => {
                let mut value = T::zero();
                for offset in 0..reduce_len {
                    let index = base + offset * layout.stride()[dim];
                    value = value.add(*values.get(index).ok_or(Error::StorageOutOfBounds)?);
                }
                output.push(value);
            }
            ReduceOp::Min | ReduceOp::Max => {
                let use_min = op == ReduceOp::Min;
                let first = *values.get(base).ok_or(Error::StorageOutOfBounds)?;
                let mut value = first;
                for offset in 1..reduce_len {
                    let index = base + offset * layout.stride()[dim];
                    value = select_extreme(
                        value,
                        *values.get(index).ok_or(Error::StorageOutOfBounds)?,
                        use_min,
                    );
                }
                output.push(value);
            }
            ReduceOp::ArgMin | ReduceOp::ArgMax => {
                unreachable!("arg reductions use arg_reduce_map")
            }
        }
    }
    Ok(output)
}

pub fn arg_reduce_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
    op: ReduceOp,
) -> Result<Vec<i64>> {
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
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates, keepdim);
        let use_min = op == ReduceOp::ArgMin;
        let mut best_index = 0usize;
        let mut best = *values.get(base).ok_or(Error::StorageOutOfBounds)?;
        for offset in 1..reduce_len {
            let index = base + offset * layout.stride()[dim];
            let candidate = *values.get(index).ok_or(Error::StorageOutOfBounds)?;
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
        output.push(best_index as i64);
    }
    Ok(output)
}

pub fn reduce_all_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    op: ReduceOp,
) -> Result<Vec<T>> {
    match op {
        ReduceOp::Sum => {
            let mut value = T::zero();
            for index in layout.strided_index() {
                value = value.add(*values.get(index).ok_or(Error::StorageOutOfBounds)?);
            }
            Ok(vec![value])
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
            let mut value = *values.get(first_index).ok_or(Error::StorageOutOfBounds)?;
            let use_min = op == ReduceOp::Min;
            for index in indices {
                value = select_extreme(
                    value,
                    *values.get(index).ok_or(Error::StorageOutOfBounds)?,
                    use_min,
                );
            }
            Ok(vec![value])
        }
        ReduceOp::ArgMin | ReduceOp::ArgMax => {
            unreachable!("arg reductions are dimension-only")
        }
    }
}

pub fn mean_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
) -> Result<Vec<T>> {
    reduce_f64_map(values, layout, dim, keepdim, false)
}

pub fn var_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
) -> Result<Vec<T>> {
    reduce_f64_map(values, layout, dim, keepdim, true)
}

fn reduce_f64_map<T: ReduceElement>(
    values: &[T],
    layout: &Layout,
    dim: usize,
    keepdim: bool,
    variance: bool,
) -> Result<Vec<T>> {
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
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let coordinates = output_coordinates(&output_dims, output_index);
        let base = input_base_index(layout, dim, &coordinates, keepdim);
        let mut sum = 0.0;
        for offset in 0..reduce_len {
            let index = base + offset * layout.stride()[dim];
            sum += values.get(index).ok_or(Error::StorageOutOfBounds)?.to_f64();
        }
        let mean = sum / reduce_len as f64;
        let value = if variance {
            let mut squared_error = 0.0;
            for offset in 0..reduce_len {
                let index = base + offset * layout.stride()[dim];
                let delta = values.get(index).ok_or(Error::StorageOutOfBounds)?.to_f64() - mean;
                squared_error += delta * delta;
            }
            squared_error / (reduce_len - 1) as f64
        } else {
            mean
        };
        output.push(T::from_f64(value));
    }
    Ok(output)
}

pub fn mean_all_map<T: ReduceElement>(values: &[T], layout: &Layout) -> Result<Vec<T>> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for index in layout.strided_index() {
        sum += values.get(index).ok_or(Error::StorageOutOfBounds)?.to_f64();
        count += 1;
    }
    if count == 0 {
        return Err(Error::EmptyReduction {
            op: "mean_all",
            dim: 0,
        });
    }
    Ok(vec![T::from_f64(sum / count as f64)])
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
