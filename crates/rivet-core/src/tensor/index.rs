use super::Tensor;
use crate::storage::Storage;
use crate::{Error, Result, Shape};

impl Tensor {
    /// Returns an overlapping sliding-window view along `dim`.
    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Result<Self> {
        self.from_shared_storage(self.layout().unfold(dim, size, step)?)
    }

    /// Reverses values along the selected dimensions into a fresh contiguous tensor.
    pub fn flip(&self, dims: &[usize]) -> Result<Self> {
        let storage = self.storage();
        let storage = Storage::flip(&storage, self.layout(), dims)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn pad_with_zeros(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        self.dim(dim)?;
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        let mut dims = self.dims().to_vec();
        dims[dim] = left;
        let left_tensor = Tensor::zeros(dims.as_slice(), self.dtype(), self.device())?;
        dims[dim] = right;
        let right_tensor = Tensor::zeros(dims.as_slice(), self.dtype(), self.device())?;
        match (left, right) {
            (0, 0) => Ok(self.clone()),
            (0, _) => Tensor::cat(&[self, &right_tensor], dim),
            (_, 0) => Tensor::cat(&[&left_tensor, self], dim),
            _ => Tensor::cat(&[&left_tensor, self, &right_tensor], dim),
        }
    }

    pub fn pad_with_same(&self, dim: usize, left: usize, right: usize) -> Result<Self> {
        self.dim(dim)?;
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        if self.elem_count() == 0 {
            return Err(Error::EmptyTensorForOp {
                op: "pad_with_same",
            });
        }

        let first = self.narrow(dim, 0, 1)?;
        let last = self.narrow(dim, self.dim(dim)? - 1, 1)?;
        let mut tensors = Vec::with_capacity(left + right + 1);
        tensors.extend(std::iter::repeat_n(&first, left));
        tensors.push(self);
        tensors.extend(std::iter::repeat_n(&last, right));
        Tensor::cat(&tensors, dim)
    }

    fn validate_index_select(&self, indexes: &Self, dim: usize) -> Result<()> {
        self.dim(dim)?;
        if indexes.rank() != 1 {
            return Err(Error::InvalidRank {
                expected: 1,
                actual: indexes.rank(),
            });
        }
        if !self.device().same_device(indexes.device()) {
            return Err(Error::DeviceMismatch);
        }
        Ok(())
    }

    pub fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self> {
        self.validate_index_select(indexes, dim)?;
        let storage = self.storage();
        let indexes_storage = indexes.storage();
        let storage = Storage::index_select(
            &storage,
            self.layout(),
            &indexes_storage,
            indexes.layout(),
            dim,
        )?;
        let mut dims = self.dims().to_vec();
        dims[dim] = indexes.dims()[0];
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    pub fn gather(&self, indexes: &Self, dim: usize) -> Result<Self> {
        self.dim(dim)?;
        if self.rank() != indexes.rank() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: indexes.dims().to_vec(),
            });
        }
        if !self.device().same_device(indexes.device()) {
            return Err(Error::DeviceMismatch);
        }
        for axis in 0..self.rank() {
            if axis != dim && indexes.dims()[axis] > self.dims()[axis] {
                return Err(Error::ShapeMismatchBinary {
                    lhs: self.dims().to_vec(),
                    rhs: indexes.dims().to_vec(),
                });
            }
        }
        let storage = self.storage();
        let indexes_storage = indexes.storage();
        let storage = Storage::gather(
            &storage,
            self.layout(),
            &indexes_storage,
            indexes.layout(),
            dim,
        )?;
        Self::from_storage(storage, indexes.shape().clone(), self.device())
    }

    pub fn embedding(&self, indexes: &Self) -> Result<Self> {
        if self.rank() != 2 || indexes.rank() != 1 {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: indexes.dims().to_vec(),
            });
        }
        self.index_select(indexes, 0)
    }

    fn validate_scatter(&self, indexes: &Self, source: &Self, dim: usize) -> Result<()> {
        self.dim(dim)?;
        if self.dtype() != source.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: source.dtype(),
            });
        }
        if !self.device().same_device(indexes.device())
            || !self.device().same_device(source.device())
        {
            return Err(Error::DeviceMismatch);
        }
        if self.rank() != source.rank() || indexes.dims() != source.dims() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: source.dims().to_vec(),
            });
        }
        for axis in 0..self.rank() {
            if axis != dim && self.dims()[axis] != source.dims()[axis] {
                return Err(Error::ShapeMismatchBinary {
                    lhs: self.dims().to_vec(),
                    rhs: source.dims().to_vec(),
                });
            }
        }
        Ok(())
    }

    fn scatter_impl(&self, indexes: &Self, source: &Self, dim: usize, add: bool) -> Result<Self> {
        self.validate_scatter(indexes, source, dim)?;
        let storage = self.storage();
        let indexes_storage = indexes.storage();
        let source_storage = source.storage();
        let storage = Storage::scatter(
            &storage,
            self.layout(),
            &indexes_storage,
            indexes.layout(),
            &source_storage,
            source.layout(),
            dim,
            add,
        )?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn scatter(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self> {
        self.scatter_impl(indexes, source, dim, false)
    }

    pub fn scatter_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self> {
        self.scatter_impl(indexes, source, dim, true)
    }

    fn validate_index_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<()> {
        self.dim(dim)?;
        if self.dtype() != source.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: source.dtype(),
            });
        }
        if !self.device().same_device(indexes.device())
            || !self.device().same_device(source.device())
        {
            return Err(Error::DeviceMismatch);
        }
        if indexes.rank() != 1
            || self.rank() != source.rank()
            || source.dims()[dim] != indexes.dims()[0]
        {
            return Err(Error::ShapeMismatchBinary {
                lhs: indexes.dims().to_vec(),
                rhs: source.dims().to_vec(),
            });
        }
        for axis in 0..self.rank() {
            if axis != dim && self.dims()[axis] != source.dims()[axis] {
                return Err(Error::ShapeMismatchBinary {
                    lhs: self.dims().to_vec(),
                    rhs: source.dims().to_vec(),
                });
            }
        }
        Ok(())
    }

    pub fn index_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self> {
        self.validate_index_add(indexes, source, dim)?;
        let storage = self.storage();
        let indexes_storage = indexes.storage();
        let source_storage = source.storage();
        let storage = Storage::index_add(
            &storage,
            self.layout(),
            &indexes_storage,
            indexes.layout(),
            &source_storage,
            source.layout(),
            dim,
        )?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn slice_scatter(&self, source: &Self, dim: usize, start: usize) -> Result<Self> {
        self.dim(dim)?;
        if self.dtype() != source.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: source.dtype(),
            });
        }
        if !self.device().same_device(source.device()) || self.rank() != source.rank() {
            return Err(Error::DeviceMismatch);
        }
        for axis in 0..self.rank() {
            if axis != dim && self.dims()[axis] != source.dims()[axis] {
                return Err(Error::ShapeMismatchBinary {
                    lhs: self.dims().to_vec(),
                    rhs: source.dims().to_vec(),
                });
            }
        }
        let source_len = source.dims()[dim];
        if start > self.dims()[dim] || source_len > self.dims()[dim] - start {
            return Err(Error::InvalidNarrow {
                dim,
                start,
                len: source_len,
                dim_size: self.dims()[dim],
            });
        }
        let before = self.narrow(dim, 0, start)?;
        let after = self.narrow(
            dim,
            start + source_len,
            self.dims()[dim] - start - source_len,
        )?;
        Tensor::cat(&[&before, source, &after], dim)
    }

    pub fn slice_scatter0(&self, source: &Self, start: usize) -> Result<Self> {
        self.slice_scatter(source, 0, start)
    }
}
