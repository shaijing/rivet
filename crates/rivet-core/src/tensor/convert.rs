use super::Tensor;
use crate::storage::Storage;
use crate::{DType, Device, Result};
use std::sync::Arc;

impl Tensor {
    /// Copies the logical tensor to `device` and returns a contiguous result.
    /// Calling this with the same logical device keeps the existing shared
    /// storage, matching the cheap-view behavior of the tensor API.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device().same_device(device) {
            return Ok(self.clone());
        }

        let storage = Storage::to_device(self.storage(), self.layout(), device)?;
        Self::from_exact_owned_storage(storage, self.shape().clone())
    }

    /// Copies the complete backing allocation and preserves this view's layout.
    pub fn copy(&self) -> Result<Self> {
        let storage = self.storage().try_clone(self.layout())?;
        Self::from_parts_checked(
            Arc::new(storage),
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
        Self::from_exact_owned_storage(storage, self.shape().clone())
    }

    /// Materializes the logical tensor into a new contiguous allocation even
    /// when the input is already contiguous.
    pub fn force_contiguous(&self) -> Result<Self> {
        let storage = Storage::copy_logical(&self.storage(), self.layout())?;
        Self::from_exact_owned_storage(storage, self.shape().clone())
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        let storage = self.storage().to_dtype(self.layout(), dtype)?;
        Self::from_exact_owned_storage(storage, self.shape().clone())
    }
}
