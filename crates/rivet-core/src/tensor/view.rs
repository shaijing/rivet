use super::Tensor;
use crate::{Error, Layout, Result, Shape};

impl Tensor {
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if self.dims().get(dim).copied() == Some(len) && start == 0 {
            return Ok(self.clone());
        }
        self.from_shared_storage(self.layout().narrow(dim, start, len)?)
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        self.from_shared_storage(self.layout().transpose(dim1, dim2)?)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        self.from_shared_storage(self.layout().permute(dims)?)
    }

    pub fn broadcast_as<S>(&self, shape: S) -> Result<Self>
    where
        S: Into<Shape>,
    {
        self.from_shared_storage(self.layout().broadcast_as(shape.into())?)
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
}
