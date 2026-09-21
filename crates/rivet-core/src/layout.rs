use crate::error::{Error, Result};
use crate::{Shape, StridedIndex};

/// A view over a flat storage allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    /// Strides and offsets are measured in elements, not bytes.
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Result<Self> {
        if shape.rank() != stride.len() {
            return Err(Error::InvalidLayout {
                rank: shape.rank(),
                stride_len: stride.len(),
            });
        }
        Ok(Self {
            shape,
            stride,
            start_offset,
        })
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn checked_elem_count(&self) -> Result<usize> {
        self.shape.checked_elem_count()
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        self.shape.is_fortran_contiguous(&self.stride)
    }

    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        self.is_contiguous().then_some((
            self.start_offset,
            self.start_offset.checked_add(self.elem_count())?,
        ))
    }

    /// Returns the inclusive range of storage elements touched by this view.
    ///
    /// Layout strides are non-negative, so the first element is always at
    /// `start_offset` and the last one is the sum of each dimension's span.
    /// Empty layouts do not touch storage and therefore return `None`.
    pub fn storage_bounds(&self) -> Option<(usize, usize)> {
        if self.elem_count() == 0 {
            return None;
        }

        let max_offset = self.max_storage_offset()?;
        Some((self.start_offset, max_offset))
    }

    /// Returns the greatest storage element offset addressed by this view.
    ///
    /// `None` means either that the layout is empty or that computing the
    /// offset overflowed `usize`.
    pub fn max_storage_offset(&self) -> Option<usize> {
        if self.elem_count() == 0 {
            return None;
        }

        self.dims().iter().zip(self.stride()).try_fold(
            self.start_offset,
            |max_offset, (&dim, &stride)| {
                let span = dim.checked_sub(1)?.checked_mul(stride)?;
                max_offset.checked_add(span)
            },
        )
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dim_size = *self.dims().get(dim).ok_or(Error::InvalidDim {
            dim,
            rank: self.shape.rank(),
        })?;
        if start > dim_size || len > dim_size - start {
            return Err(Error::InvalidNarrow {
                dim,
                start,
                len,
                dim_size,
            });
        }
        let offset = self
            .start_offset
            .checked_add(
                start
                    .checked_mul(self.stride[dim])
                    .ok_or(Error::StorageOutOfBounds)?,
            )
            .ok_or(Error::StorageOutOfBounds)?;
        let mut dims = self.dims().to_vec();
        dims[dim] = len;
        Self::new(Shape::from(dims), self.stride.clone(), offset)
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 >= self.shape.rank() {
            return Err(Error::InvalidDim {
                dim: dim1,
                rank: self.shape.rank(),
            });
        }
        if dim2 >= self.shape.rank() {
            return Err(Error::InvalidDim {
                dim: dim2,
                rank: self.shape.rank(),
            });
        }
        let mut dims = self.dims().to_vec();
        let mut stride = self.stride.clone();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Self::new(Shape::from(dims), stride, self.start_offset)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let rank = self.shape.rank();
        let valid = dims.len() == rank && dims.iter().all(|&dim| dim < rank) && {
            let mut seen = vec![false; rank];
            dims.iter().all(|&dim| {
                if seen[dim] {
                    false
                } else {
                    seen[dim] = true;
                    true
                }
            })
        };
        if !valid {
            return Err(Error::InvalidPermutation(dims.to_vec()));
        }
        let shape = Shape::from(dims.iter().map(|&dim| self.dims()[dim]).collect::<Vec<_>>());
        let stride = dims.iter().map(|&dim| self.stride[dim]).collect::<Vec<_>>();
        Self::new(shape, stride, self.start_offset)
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape.rank() {
            return Err(Error::InvalidBroadcast {
                lhs: self.dims().to_vec(),
                rhs: shape.dims().to_vec(),
            });
        }
        let leading = shape.rank() - self.shape.rank();
        let mut stride = vec![0; leading];
        for ((&source_dim, &source_stride), &target_dim) in self
            .dims()
            .iter()
            .zip(&self.stride)
            .zip(&shape.dims()[leading..])
        {
            if source_dim == target_dim {
                stride.push(source_stride);
            } else if source_dim == 1 {
                stride.push(0);
            } else {
                return Err(Error::InvalidBroadcast {
                    lhs: self.dims().to_vec(),
                    rhs: shape.dims().to_vec(),
                });
            }
        }
        Self::new(shape, stride, self.start_offset)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dim_size = *self.dims().get(dim).ok_or(Error::InvalidDim {
            dim,
            rank: self.shape.rank(),
        })?;
        if dim_size != 1 {
            return Ok(self.clone());
        }
        let mut dims = self.dims().to_vec();
        let mut stride = self.stride.clone();
        dims.remove(dim);
        stride.remove(dim);
        Self::new(Shape::from(dims), stride, self.start_offset)
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        if dim > self.shape.rank() {
            return Err(Error::InvalidDim {
                dim,
                rank: self.shape.rank() + 1,
            });
        }
        let mut dims = self.dims().to_vec();
        let mut stride = self.stride.clone();
        dims.insert(dim, 1);
        let inserted_stride = stride.get(dim).copied().unwrap_or(1);
        stride.insert(dim, inserted_stride);
        Self::new(Shape::from(dims), stride, self.start_offset)
    }

    /// Creates an overlapping sliding-window view along `dim`.
    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Result<Self> {
        let dim_size = *self.dims().get(dim).ok_or(Error::InvalidDim {
            dim,
            rank: self.shape.rank(),
        })?;
        if step == 0 || size > dim_size {
            return Err(Error::InvalidUnfold {
                dim,
                size,
                step,
                dim_size,
            });
        }
        let window_count = (dim_size - size) / step + 1;
        let mut dims = self.dims().to_vec();
        dims[dim] = window_count;
        dims.push(size);
        let mut stride = self.stride.clone();
        stride[dim] = stride[dim]
            .checked_mul(step)
            .ok_or(Error::StorageOutOfBounds)?;
        stride.push(self.stride[dim]);
        Self::new(Shape::from(dims), stride, self.start_offset)
    }

    pub fn strided_index(&self) -> StridedIndex<'_> {
        StridedIndex::new(self.dims(), self.stride(), self.start_offset())
    }
}
