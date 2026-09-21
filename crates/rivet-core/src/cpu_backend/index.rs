use super::buffer::{AlignedBuffer, AlignedBufferBuilder};
use super::storage::{aligned, CpuStorage};
use super::utils::{copy_logical, BinaryElement, ValidatedValues};
use crate::ops::BinaryOp;
use crate::{Error, Layout, Result, Shape};

fn aligned_from_iter<T, I>(iter: I) -> Result<AlignedBuffer<T>>
where
    I: ExactSizeIterator<Item = Result<T>>,
{
    let mut builder = AlignedBufferBuilder::new(iter.len())?;
    for value in iter {
        builder.write_next(value?)?;
    }
    builder.finish()
}

/// Integer values accepted by indexing operations. Negative values are
/// rejected rather than being interpreted as Python-style offsets.
pub trait IndexElement: Copy + Send + Sync + 'static {
    fn to_index(self, op: &'static str) -> Result<usize>;
}

macro_rules! impl_unsigned_index {
    ($ty:ty) => {
        impl IndexElement for $ty {
            fn to_index(self, op: &'static str) -> Result<usize> {
                usize::try_from(self).map_err(|_| Error::InvalidIndex {
                    op,
                    index: usize::MAX,
                    size: usize::MAX,
                })
            }
        }
    };
}

macro_rules! impl_signed_index {
    ($ty:ty) => {
        impl IndexElement for $ty {
            fn to_index(self, op: &'static str) -> Result<usize> {
                if self < 0 {
                    return Err(Error::NegativeIndex {
                        op,
                        value: self as i64,
                    });
                }
                usize::try_from(self as i64).map_err(|_| Error::InvalidIndex {
                    op,
                    index: usize::MAX,
                    size: usize::MAX,
                })
            }
        }
    };
}

impl_unsigned_index!(u8);
impl_unsigned_index!(u32);
impl_signed_index!(i16);
impl_signed_index!(i32);
impl_signed_index!(i64);

fn coordinates(dims: &[usize], mut linear: usize) -> Vec<usize> {
    let mut output = vec![0; dims.len()];
    for axis in (0..dims.len()).rev() {
        output[axis] = linear % dims[axis];
        linear /= dims[axis];
    }
    output
}

fn row_major_index(coordinates: &[usize], dims: &[usize]) -> Result<usize> {
    coordinates
        .iter()
        .zip(dims)
        .try_fold(0usize, |index, (&coordinate, &dim)| {
            index
                .checked_mul(dim)
                .and_then(|index| index.checked_add(coordinate))
                .ok_or(Error::StorageOutOfBounds)
        })
}

fn physical_index(layout: &Layout, coordinates: &[usize]) -> Result<usize> {
    let mut index = layout.start_offset();
    for ((&coordinate, &dim), &stride) in coordinates.iter().zip(layout.dims()).zip(layout.stride())
    {
        if coordinate >= dim {
            return Err(Error::StorageOutOfBounds);
        }
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

fn read_index<I: IndexElement>(
    indexes: &ValidatedValues<'_, I>,
    indexes_layout: &Layout,
    coordinates: &[usize],
    size: usize,
    op: &'static str,
) -> Result<usize> {
    let physical = physical_index(indexes_layout, coordinates)?;
    let index = indexes.read(physical);
    let index = index.to_index(op)?;
    if index >= size {
        return Err(Error::InvalidIndex { op, index, size });
    }
    Ok(index)
}

pub(crate) fn flip_map<T: Copy>(
    values: &[T],
    layout: &Layout,
    dims: &[usize],
) -> Result<AlignedBuffer<T>> {
    let values = ValidatedValues::new(values, layout)?;
    let output_len = layout.checked_elem_count()?;
    let mut reverse = vec![false; layout.dims().len()];
    for &dim in dims {
        if dim >= reverse.len() {
            return Err(Error::InvalidDim {
                dim,
                rank: reverse.len(),
            });
        }
        reverse[dim] = !reverse[dim];
    }

    aligned_from_iter((0..output_len).map(|linear| {
        let mut coordinates = coordinates(layout.dims(), linear);
        for (axis, coordinate) in coordinates.iter_mut().enumerate() {
            if reverse[axis] {
                *coordinate = layout.dims()[axis] - 1 - *coordinate;
            }
        }
        let physical = physical_index(layout, &coordinates)?;
        Ok(values.read(physical))
    }))
}

pub fn gather_map<T: Copy, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    dim: usize,
) -> Result<AlignedBuffer<T>> {
    if values_layout.dims().len() != indexes_layout.dims().len()
        || dim >= values_layout.dims().len()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: values_layout.dims().to_vec(),
            rhs: indexes_layout.dims().to_vec(),
        });
    }

    let values = ValidatedValues::new(values, values_layout)?;
    let indexes = ValidatedValues::new(indexes, indexes_layout)?;
    let output_len = indexes_layout.checked_elem_count()?;

    aligned_from_iter((0..output_len).map(|linear| {
        let coordinates = coordinates(indexes_layout.dims(), linear);
        let index = read_index(
            &indexes,
            indexes_layout,
            &coordinates,
            values_layout.dims()[dim],
            "gather",
        )?;
        let mut source_coordinates = coordinates;
        source_coordinates[dim] = index;
        let physical = physical_index(values_layout, &source_coordinates)?;
        Ok(values.read(physical))
    }))
}

pub fn index_select_map<T: Copy, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    dim: usize,
) -> Result<AlignedBuffer<T>> {
    if indexes_layout.dims().len() != 1 || dim >= values_layout.dims().len() {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: indexes_layout.dims().len(),
        });
    }
    let mut output_dims = values_layout.dims().to_vec();
    output_dims[dim] = indexes_layout.dims()[0];
    let values = ValidatedValues::new(values, values_layout)?;
    let indexes = ValidatedValues::new(indexes, indexes_layout)?;
    let output_len = Shape::from(output_dims.clone()).checked_elem_count()?;
    aligned_from_iter((0..output_len).map(|linear| {
        let coordinates = coordinates(&output_dims, linear);
        let index = read_index(
            &indexes,
            indexes_layout,
            &[coordinates[dim]],
            values_layout.dims()[dim],
            "index_select",
        )?;
        let mut source_coordinates = coordinates;
        source_coordinates[dim] = index;
        let physical = physical_index(values_layout, &source_coordinates)?;
        Ok(values.read(physical))
    }))
}

impl CpuStorage {
    pub(crate) fn flip(&self, layout: &Layout, dims: &[usize]) -> Result<Self> {
        macro_rules! dispatch {
            ($variant:ident, $ty:ty) => {
                if let Self::$variant(values) = self {
                    Ok(Self::$variant(aligned(flip_map(values, layout, dims)?)?))
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
            ($values_variant:ident, $values:expr) => {
                match indexes {
                    Self::U8(ids) => Ok(Self::$values_variant(aligned(gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$values_variant(aligned(gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$values_variant(aligned(gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$values_variant(aligned(gather_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$values_variant(aligned(gather_map(
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
                    Self::U8(ids) => Ok(Self::$values_variant(aligned(index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$values_variant(aligned(index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$values_variant(aligned(index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$values_variant(aligned(index_select_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        dim,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$values_variant(aligned(index_select_map(
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
                    Self::U8(ids) => Ok(Self::$variant(aligned(scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$variant(aligned(scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$variant(aligned(scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$variant(aligned(scatter_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                        add,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$variant(aligned(scatter_map(
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
            (Self::U8(values), Self::U8(source)) => dispatch_indexes!(U8, values, source),
            (Self::U32(values), Self::U32(source)) => dispatch_indexes!(U32, values, source),
            (Self::I16(values), Self::I16(source)) => dispatch_indexes!(I16, values, source),
            (Self::I32(values), Self::I32(source)) => dispatch_indexes!(I32, values, source),
            (Self::I64(values), Self::I64(source)) => dispatch_indexes!(I64, values, source),
            (Self::BF16(values), Self::BF16(source)) => dispatch_indexes!(BF16, values, source),
            (Self::F16(values), Self::F16(source)) => dispatch_indexes!(F16, values, source),
            (Self::F32(values), Self::F32(source)) => dispatch_indexes!(F32, values, source),
            (Self::F64(values), Self::F64(source)) => dispatch_indexes!(F64, values, source),
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
                    Self::U8(ids) => Ok(Self::$variant(aligned(index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::U32(ids) => Ok(Self::$variant(aligned(index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::I16(ids) => Ok(Self::$variant(aligned(index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::I32(ids) => Ok(Self::$variant(aligned(index_add_map(
                        $values,
                        values_layout,
                        ids,
                        indexes_layout,
                        $source,
                        source_layout,
                        dim,
                    )?)?)),
                    Self::I64(ids) => Ok(Self::$variant(aligned(index_add_map(
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
            (Self::U8(values), Self::U8(source)) => dispatch_indexes!(U8, values, source),
            (Self::U32(values), Self::U32(source)) => dispatch_indexes!(U32, values, source),
            (Self::I16(values), Self::I16(source)) => dispatch_indexes!(I16, values, source),
            (Self::I32(values), Self::I32(source)) => dispatch_indexes!(I32, values, source),
            (Self::I64(values), Self::I64(source)) => dispatch_indexes!(I64, values, source),
            (Self::BF16(values), Self::BF16(source)) => dispatch_indexes!(BF16, values, source),
            (Self::F16(values), Self::F16(source)) => dispatch_indexes!(F16, values, source),
            (Self::F32(values), Self::F32(source)) => dispatch_indexes!(F32, values, source),
            (Self::F64(values), Self::F64(source)) => dispatch_indexes!(F64, values, source),
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
}

pub fn scatter_map<T: Copy + BinaryElement, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    source: &[T],
    source_layout: &Layout,
    dim: usize,
    add: bool,
) -> Result<AlignedBuffer<T>> {
    if indexes_layout.dims() != source_layout.dims()
        || values_layout.dims().len() != source_layout.dims().len()
        || dim >= values_layout.dims().len()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: values_layout.dims().to_vec(),
            rhs: source_layout.dims().to_vec(),
        });
    }
    let indexes = ValidatedValues::new(indexes, indexes_layout)?;
    let source = ValidatedValues::new(source, source_layout)?;
    let mut output = copy_logical(values, values_layout)?;
    let source_len = source_layout.checked_elem_count()?;
    for linear in 0..source_len {
        let coordinates = coordinates(source_layout.dims(), linear);
        let index = read_index(
            &indexes,
            indexes_layout,
            &coordinates,
            values_layout.dims()[dim],
            if add { "scatter_add" } else { "scatter" },
        )?;
        let source_physical = physical_index(source_layout, &coordinates)?;
        let mut target_coordinates = coordinates;
        target_coordinates[dim] = index;
        let target = row_major_index(&target_coordinates, values_layout.dims())?;
        let source_value = source.read(source_physical);
        if add {
            output.as_mut_slice()[target] =
                T::apply(BinaryOp::Add, output.as_slice()[target], source_value)?;
        } else {
            output.as_mut_slice()[target] = source_value;
        }
    }
    Ok(output)
}

pub fn index_add_map<T: Copy + BinaryElement, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    source: &[T],
    source_layout: &Layout,
    dim: usize,
) -> Result<AlignedBuffer<T>> {
    if indexes_layout.dims().len() != 1
        || dim >= values_layout.dims().len()
        || values_layout.dims().len() != source_layout.dims().len()
        || source_layout.dims()[dim] != indexes_layout.dims()[0]
        || (0..values_layout.dims().len())
            .any(|axis| axis != dim && values_layout.dims()[axis] != source_layout.dims()[axis])
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: values_layout.dims().to_vec(),
            rhs: source_layout.dims().to_vec(),
        });
    }
    let indexes = ValidatedValues::new(indexes, indexes_layout)?;
    let source = ValidatedValues::new(source, source_layout)?;
    let mut output = copy_logical(values, values_layout)?;
    let source_len = source_layout.checked_elem_count()?;
    for linear in 0..source_len {
        let coordinates = coordinates(source_layout.dims(), linear);
        let index = read_index(
            &indexes,
            indexes_layout,
            &[coordinates[dim]],
            values_layout.dims()[dim],
            "index_add",
        )?;
        let source_physical = physical_index(source_layout, &coordinates)?;
        let mut target_coordinates = coordinates;
        target_coordinates[dim] = index;
        let target = row_major_index(&target_coordinates, values_layout.dims())?;
        let source_value = source.read(source_physical);
        output.as_mut_slice()[target] =
            T::apply(BinaryOp::Add, output.as_slice()[target], source_value)?;
    }
    Ok(output)
}
