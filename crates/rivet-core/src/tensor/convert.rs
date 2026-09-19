use super::Tensor;
use crate::storage::Storage;
use crate::{DType, Result};
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Copies the complete backing allocation and preserves this view's layout.
    pub fn copy(&self) -> Result<Self> {
        let storage = self.storage().try_clone(self.layout())?;
        Self::from_parts(
            Arc::new(RwLock::new(storage)),
            self.layout().clone(),
            self.dtype(),
            self.device().clone(),
        )
    }

    /// Returns this handle for contiguous tensors, otherwise materializes the
    /// logical row-major order into exactly-sized storage.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let storage = Storage::copy_logical(&self.storage(), self.layout())?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        let storage = self.storage().to_dtype(self.layout(), dtype)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }
}
