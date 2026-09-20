use super::utils::{BinaryElement, copy_logical};
use crate::ops::BinaryOp;
use crate::{Error, Layout, Result};

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

fn row_major_index(coordinates: &[usize], dims: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(dims)
        .fold(0, |index, (&coordinate, &dim)| index * dim + coordinate)
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
    indexes: &[I],
    indexes_layout: &Layout,
    coordinates: &[usize],
    size: usize,
    op: &'static str,
) -> Result<usize> {
    let physical = physical_index(indexes_layout, coordinates)?;
    let index = *indexes.get(physical).ok_or(Error::StorageOutOfBounds)?;
    let index = index.to_index(op)?;
    if index >= size {
        return Err(Error::InvalidIndex { op, index, size });
    }
    Ok(index)
}

pub fn flip_map<T: Copy>(values: &[T], layout: &Layout, dims: &[usize]) -> Result<Vec<T>> {
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

    let mut output = Vec::with_capacity(layout.elem_count());
    for linear in 0..layout.elem_count() {
        let mut coordinates = coordinates(layout.dims(), linear);
        for (axis, coordinate) in coordinates.iter_mut().enumerate() {
            if reverse[axis] {
                *coordinate = layout.dims()[axis] - 1 - *coordinate;
            }
        }
        let physical = physical_index(layout, &coordinates)?;
        output.push(*values.get(physical).ok_or(Error::StorageOutOfBounds)?);
    }
    Ok(output)
}

pub fn gather_map<T: Copy, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    dim: usize,
) -> Result<Vec<T>> {
    if values_layout.dims().len() != indexes_layout.dims().len()
        || dim >= values_layout.dims().len()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: values_layout.dims().to_vec(),
            rhs: indexes_layout.dims().to_vec(),
        });
    }

    let mut output = Vec::with_capacity(indexes_layout.elem_count());
    for linear in 0..indexes_layout.elem_count() {
        let coordinates = coordinates(indexes_layout.dims(), linear);
        let index = read_index(
            indexes,
            indexes_layout,
            &coordinates,
            values_layout.dims()[dim],
            "gather",
        )?;
        let mut source_coordinates = coordinates;
        source_coordinates[dim] = index;
        let physical = physical_index(values_layout, &source_coordinates)?;
        output.push(*values.get(physical).ok_or(Error::StorageOutOfBounds)?);
    }
    Ok(output)
}

pub fn index_select_map<T: Copy, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    dim: usize,
) -> Result<Vec<T>> {
    if indexes_layout.dims().len() != 1 || dim >= values_layout.dims().len() {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: indexes_layout.dims().len(),
        });
    }
    let mut output_dims = values_layout.dims().to_vec();
    output_dims[dim] = indexes_layout.dims()[0];
    let output_len = output_dims.iter().product::<usize>();
    let mut output = Vec::with_capacity(output_len);
    for linear in 0..output_len {
        let coordinates = coordinates(&output_dims, linear);
        let index = read_index(
            indexes,
            indexes_layout,
            &[coordinates[dim]],
            values_layout.dims()[dim],
            "index_select",
        )?;
        let mut source_coordinates = coordinates;
        source_coordinates[dim] = index;
        let physical = physical_index(values_layout, &source_coordinates)?;
        output.push(*values.get(physical).ok_or(Error::StorageOutOfBounds)?);
    }
    Ok(output)
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
) -> Result<Vec<T>> {
    if indexes_layout.dims() != source_layout.dims()
        || values_layout.dims().len() != source_layout.dims().len()
        || dim >= values_layout.dims().len()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: values_layout.dims().to_vec(),
            rhs: source_layout.dims().to_vec(),
        });
    }
    let mut output = copy_logical(values, values_layout)?;
    for linear in 0..source_layout.elem_count() {
        let coordinates = coordinates(source_layout.dims(), linear);
        let index = read_index(
            indexes,
            indexes_layout,
            &coordinates,
            values_layout.dims()[dim],
            if add { "scatter_add" } else { "scatter" },
        )?;
        let source_physical = physical_index(source_layout, &coordinates)?;
        let mut target_coordinates = coordinates;
        target_coordinates[dim] = index;
        let target = row_major_index(&target_coordinates, values_layout.dims());
        let source_value = *source
            .get(source_physical)
            .ok_or(Error::StorageOutOfBounds)?;
        if add {
            output[target] = T::apply(BinaryOp::Add, output[target], source_value)?;
        } else {
            output[target] = source_value;
        }
    }
    Ok(output)
}

pub fn scatter_in_place<T: Copy + BinaryElement, I: IndexElement>(
    values: &mut [T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    source: &[T],
    source_layout: &Layout,
    dim: usize,
    add: bool,
) -> Result<()> {
    if indexes_layout.dims() != source_layout.dims()
        || values_layout.dims().len() != source_layout.dims().len()
        || dim >= values_layout.dims().len()
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: values_layout.dims().to_vec(),
            rhs: source_layout.dims().to_vec(),
        });
    }
    for linear in 0..source_layout.elem_count() {
        let coordinates = coordinates(source_layout.dims(), linear);
        let index = read_index(
            indexes,
            indexes_layout,
            &coordinates,
            values_layout.dims()[dim],
            if add { "scatter_add" } else { "scatter" },
        )?;
        let source_physical = physical_index(source_layout, &coordinates)?;
        let mut target_coordinates = coordinates;
        target_coordinates[dim] = index;
        let target = physical_index(values_layout, &target_coordinates)?;
        let source_value = *source
            .get(source_physical)
            .ok_or(Error::StorageOutOfBounds)?;
        if add {
            values[target] = T::apply(BinaryOp::Add, values[target], source_value)?;
        } else {
            values[target] = source_value;
        }
    }
    Ok(())
}

pub fn index_add_map<T: Copy + BinaryElement, I: IndexElement>(
    values: &[T],
    values_layout: &Layout,
    indexes: &[I],
    indexes_layout: &Layout,
    source: &[T],
    source_layout: &Layout,
    dim: usize,
) -> Result<Vec<T>> {
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
    let mut output = copy_logical(values, values_layout)?;
    for linear in 0..source_layout.elem_count() {
        let coordinates = coordinates(source_layout.dims(), linear);
        let index = read_index(
            indexes,
            indexes_layout,
            &[coordinates[dim]],
            values_layout.dims()[dim],
            "index_add",
        )?;
        let source_physical = physical_index(source_layout, &coordinates)?;
        let mut target_coordinates = coordinates;
        target_coordinates[dim] = index;
        let target = row_major_index(&target_coordinates, values_layout.dims());
        let source_value = *source
            .get(source_physical)
            .ok_or(Error::StorageOutOfBounds)?;
        output[target] = T::apply(BinaryOp::Add, output[target], source_value)?;
    }
    Ok(output)
}
