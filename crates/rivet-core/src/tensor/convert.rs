use super::Tensor;
use crate::storage::Storage;
use crate::{DType, Device, Result};
use std::sync::Arc;

impl Tensor {
    /// Waits for all asynchronous work associated with this tensor's device.
    /// CPU tensors are already complete, so this is a no-op on CPU.
    pub fn synchronize(&self) -> Result<()> {
        self.device().synchronize()
    }

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

    /// Runs the fused U8 NHWC -> normalized F32 NCHW image kernel on a CUDA
    /// tensor. `scale` and `bias` are expanded per-channel affine parameters;
    /// the result is allocated directly in its final contiguous CUDA storage.
    #[cfg(feature = "cuda")]
    pub fn cuda_normalize_u8_nhwc_to_nchw_f32(&self, scale: &[f32], bias: &[f32]) -> Result<Self> {
        use crate::cuda_backend::{CudaStorage, CudaStorageView};

        if self.dtype() != DType::U8 {
            return Err(crate::Error::UnexpectedDType {
                expected: DType::U8,
                actual: self.dtype(),
            });
        }
        if self.dims().len() != 4 {
            return Err(crate::Error::InvalidRank {
                expected: 4,
                actual: self.dims().len(),
            });
        }
        if !self.is_contiguous() {
            return Err(crate::Error::UnsupportedCudaOp {
                op: "cuda_normalize_u8_nhwc_to_nchw_f32 requires contiguous input",
            });
        }
        let [batch, height, width, channels] = [
            self.dims()[0],
            self.dims()[1],
            self.dims()[2],
            self.dims()[3],
        ];
        if scale.len() != channels || bias.len() != channels {
            return Err(crate::Error::ShapeMismatch {
                expected: channels,
                actual: scale.len().min(bias.len()),
            });
        }
        let (device, input) = match (self.device(), self.storage()) {
            (Device::Cuda(device), Storage::Cuda(input)) => (device, input),
            _ => return Err(crate::Error::DeviceMismatch),
        };
        let input_view = input
            .data
            .view(self.layout().start_offset(), self.elem_count())?;
        let CudaStorageView::U8(input_view) = input_view else {
            return Err(crate::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };
        let mut output = CudaStorage::zeros(input.device_handle(), DType::F32, self.elem_count())?;
        crate::cuda_backend::normalize_u8_nhwc_to_nchw_f32(
            device,
            &CudaStorageView::U8(input_view),
            &mut output.data,
            [batch, height, width, channels],
            scale,
            bias,
        )?;
        let shape = vec![batch, channels, height, width];
        Self::from_exact_owned_storage(Storage::Cuda(output), crate::Shape::from(shape))
    }

    /// Runs one batch-native RGB image kernel on CUDA: optional crop/resize
    /// and flips, per-channel normalization, and NHWC-to-NCHW materialization.
    /// Each sample has eight U32 parameters in `[x, y, width, height,
    /// source_reverse_x, source_reverse_y, output_flip_x, output_flip_y]` order.
    #[cfg(feature = "cuda")]
    pub fn cuda_augment_normalize_u8_nhwc_to_nchw_f32(
        &self,
        output_height: usize,
        output_width: usize,
        crop_params: &[u32],
        filter: i32,
        scale: &[f32],
        bias: &[f32],
    ) -> Result<Self> {
        use crate::cuda_backend::{CudaStorage, CudaStorageView};

        if self.dtype() != DType::U8 {
            return Err(crate::Error::UnexpectedDType {
                expected: DType::U8,
                actual: self.dtype(),
            });
        }
        if self.dims().len() != 4 {
            return Err(crate::Error::InvalidRank {
                expected: 4,
                actual: self.dims().len(),
            });
        }
        if !self.is_contiguous() {
            return Err(crate::Error::UnsupportedCudaOp {
                op: "cuda_augment_normalize_u8_nhwc_to_nchw_f32 requires contiguous input",
            });
        }
        let [batch, input_height, input_width, channels] = [
            self.dims()[0],
            self.dims()[1],
            self.dims()[2],
            self.dims()[3],
        ];
        if channels != 3 {
            return Err(crate::Error::UnsupportedCudaOp {
                op: "cuda image augmentation requires three-channel RGB input",
            });
        }
        if input_height == 0 || input_width == 0 {
            return Err(crate::Error::UnsupportedCudaOp {
                op: "cuda image augmentation requires non-empty input dimensions",
            });
        }
        if output_height == 0 || output_width == 0 {
            return Err(crate::Error::UnsupportedCudaOp {
                op: "CUDA image augmentation output dimensions must be nonzero",
            });
        }
        let expected_params = batch
            .checked_mul(8)
            .ok_or(crate::Error::ShapeElementCountOverflow)?;
        if crop_params.len() != expected_params {
            return Err(crate::Error::ShapeMismatch {
                expected: expected_params,
                actual: crop_params.len(),
            });
        }
        if !(0..=3).contains(&filter) {
            return Err(crate::Error::UnsupportedCudaOp {
                op: "cuda image augmentation interpolation mode is invalid",
            });
        }
        for params in crop_params.chunks_exact(8) {
            let (x, y, width, height) = (params[0], params[1], params[2], params[3]);
            if width == 0
                || height == 0
                || x.checked_add(width)
                    .is_none_or(|end| end as usize > input_width)
                || y.checked_add(height)
                    .is_none_or(|end| end as usize > input_height)
                || params[4..8].iter().any(|flag| *flag > 1)
            {
                return Err(crate::Error::UnsupportedCudaOp {
                    op: "cuda image augmentation crop parameters are out of bounds",
                });
            }
        }
        if scale.len() != 3 || bias.len() != 3 {
            return Err(crate::Error::ShapeMismatch {
                expected: 3,
                actual: scale.len().min(bias.len()),
            });
        }
        let (device, input) = match (self.device(), self.storage()) {
            (Device::Cuda(device), Storage::Cuda(input)) => (device, input),
            _ => return Err(crate::Error::DeviceMismatch),
        };
        let input_view = input
            .data
            .view(self.layout().start_offset(), self.elem_count())?;
        let CudaStorageView::U8(input_view) = input_view else {
            return Err(crate::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };
        let output_len = batch
            .checked_mul(3)
            .and_then(|size| size.checked_mul(output_height))
            .and_then(|size| size.checked_mul(output_width))
            .ok_or(crate::Error::ShapeElementCountOverflow)?;
        let mut output = CudaStorage::zeros(input.device_handle(), DType::F32, output_len)?;
        crate::cuda_backend::vision_augment_normalize_u8_nhwc_to_nchw_f32(
            device,
            &CudaStorageView::U8(input_view),
            &mut output.data,
            [batch, input_height, input_width, channels],
            [batch, 3, output_height, output_width],
            crop_params,
            filter,
            scale,
            bias,
        )?;
        Self::from_exact_owned_storage(
            Storage::Cuda(output),
            crate::Shape::from(vec![batch, 3, output_height, output_width]),
        )
    }
}
