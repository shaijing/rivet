use super::Tensor;
use crate::{Error, Layout, Result, Shape};

impl Tensor {
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if self.dims().get(dim).copied() == Some(len) && start == 0 {
            return Ok(self.clone());
        }
        self.from_shared_storage(self.layout().narrow(dim, start, len)?)
    }

    /// Returns the slice at index `i` on the first dimension.
    pub fn get(&self, i: usize) -> Result<Self> {
        if self.rank() == 0 {
            return Ok(self.clone());
        }
        self.narrow(0, i, 1)?.reshape(&self.dims()[1..])
    }

    /// Returns the slice at `index` on `dim`, removing that dimension.
    pub fn get_on_dim(&self, dim: usize, index: usize) -> Result<Self> {
        self.narrow(dim, index, 1)?.squeeze(dim)
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        self.from_shared_storage(self.layout().transpose(dim1, dim2)?)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        self.from_shared_storage(self.layout().permute(dims)?)
    }

    /// Swaps the last two dimensions.
    pub fn t(&self) -> Result<Self> {
        if self.rank() < 2 {
            return Err(Error::InvalidRank {
                expected: 2,
                actual: self.rank(),
            });
        }
        self.transpose(self.rank() - 2, self.rank() - 1)
    }

    pub fn broadcast_as<S>(&self, shape: S) -> Result<Self>
    where
        S: Into<Shape>,
    {
        self.from_shared_storage(self.layout().broadcast_as(shape.into())?)
    }

    /// Inserts broadcast dimensions on the left of the current shape.
    pub fn broadcast_left<S>(&self, left_shape: S) -> Result<Self>
    where
        S: Into<Shape>,
    {
        let mut dims = left_shape.into().into_dims();
        dims.extend_from_slice(self.dims());
        self.broadcast_as(dims)
    }

    /// Alias for [`Tensor::broadcast_as`].
    pub fn expand<S>(&self, shape: S) -> Result<Self>
    where
        S: Into<Shape>,
    {
        self.broadcast_as(shape)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let layout = self.layout().squeeze(dim)?;
        if layout == *self.layout() {
            Ok(self.clone())
        } else {
            self.from_shared_storage(layout)
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        self.from_shared_storage(self.layout().unsqueeze(dim)?)
    }

    pub fn reshape<S>(&self, shape: S) -> Result<Self>
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            return Err(Error::InvalidReshape {
                from: self.dims().to_vec(),
                to: shape.dims().to_vec(),
            });
        }
        if self.is_contiguous() {
            return self.from_shared_storage(Layout::contiguous_with_offset(
                shape,
                self.layout().start_offset(),
            ));
        }
        self.contiguous()?.reshape(shape)
    }

    /// Splits a dimension into at most `chunks` views with balanced sizes.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        if chunks == 0 {
            return Err(Error::InvalidChunkCount { chunks });
        }
        let size = self.dim(dim)?;
        if size < chunks {
            return (0..size).map(|index| self.narrow(dim, index, 1)).collect();
        }

        let chunk_size = size / chunks;
        let additional = size % chunks;
        let mut output = Vec::with_capacity(chunks);
        let mut offset = 0usize;
        for index in 0..chunks {
            let len = chunk_size + usize::from(index < additional);
            output.push(self.narrow(dim, offset, len)?);
            offset += len;
        }
        Ok(output)
    }

    /// Repeats the tensor according to per-dimension repeat counts.
    pub fn repeat<S>(&self, repeats: S) -> Result<Self>
    where
        S: Into<Shape>,
    {
        let repeats = repeats.into().into_dims();
        let mut input = if self.rank() < repeats.len() {
            let mut shape = vec![1; repeats.len() - self.rank()];
            shape.extend_from_slice(self.dims());
            self.reshape(shape)?
        } else {
            self.clone()
        };

        for (dim, repeat) in repeats.into_iter().enumerate() {
            input = match repeat {
                0 => input.narrow(dim, 0, 0)?,
                1 => input,
                repeat => {
                    let copies = vec![&input; repeat];
                    Self::cat(&copies, dim)?
                }
            };
        }
        Ok(input)
    }

    /// Rolls values along one dimension, wrapping values that cross an edge.
    pub fn roll(&self, shift: i32, dim: usize) -> Result<Self> {
        let size = self.dim(dim)?;
        if size == 0 {
            return Ok(self.clone());
        }
        let shift = shift.rem_euclid(size as i32) as usize;
        if shift == 0 {
            return Ok(self.clone());
        }
        let first = self.narrow(dim, 0, size - shift)?;
        let second = self.narrow(dim, size - shift, shift)?;
        Self::cat(&[&second, &first], dim)
    }

    fn flatten_range(&self, start: Option<usize>, end: Option<usize>) -> Result<Self> {
        if self.rank() == 0 {
            return self.reshape(1usize);
        }
        let start = start.unwrap_or(0);
        let end = end.unwrap_or(self.rank() - 1);
        self.dim(start)?;
        self.dim(end)?;
        if start >= end {
            return Ok(self.clone());
        }

        let mut dims = self.dims()[..start].to_vec();
        dims.push(self.dims()[start..=end].iter().product());
        dims.extend_from_slice(&self.dims()[end + 1..]);
        self.reshape(dims)
    }

    /// Flattens dimensions from `start_dim` through `end_dim`.
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        self.flatten_range(Some(start_dim), Some(end_dim))
    }

    /// Flattens dimensions from the first dimension through `end_dim`.
    pub fn flatten_to(&self, end_dim: usize) -> Result<Self> {
        self.flatten_range(None, Some(end_dim))
    }

    /// Flattens dimensions from `start_dim` through the last dimension.
    pub fn flatten_from(&self, start_dim: usize) -> Result<Self> {
        self.flatten_range(Some(start_dim), None)
    }
}
