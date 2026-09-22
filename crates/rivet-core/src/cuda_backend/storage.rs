use std::fmt::Display;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr};
use half::{bf16, f16};

use crate::backend::BackendStorage;
use crate::{DType, Error, Layout, Result};

use super::device::CudaDevice;

/// Typed CUDA allocation storage. Keeping the dtype in the enum preserves
/// static dispatch once an operation selects a concrete element type.
#[derive(Debug, Clone)]
pub enum CudaStorageSlice {
    U8(CudaSlice<u8>),
    U32(CudaSlice<u32>),
    I16(CudaSlice<i16>),
    I32(CudaSlice<i32>),
    I64(CudaSlice<i64>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::I16(data) => data.len(),
            Self::I32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::BF16(data) => data.len(),
            Self::F16(data) => data.len(),
            Self::F32(data) => data.len(),
            Self::F64(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn zeros(stream: &Arc<CudaStream>, dtype: DType, len: usize) -> Result<Self> {
        match dtype {
            DType::U8 => stream
                .alloc_zeros(len)
                .map(Self::U8)
                .map_err(|error| cuda_error("zeros", error)),
            DType::U32 => stream
                .alloc_zeros(len)
                .map(Self::U32)
                .map_err(|error| cuda_error("zeros", error)),
            DType::I16 => stream
                .alloc_zeros(len)
                .map(Self::I16)
                .map_err(|error| cuda_error("zeros", error)),
            DType::I32 => stream
                .alloc_zeros(len)
                .map(Self::I32)
                .map_err(|error| cuda_error("zeros", error)),
            DType::I64 => stream
                .alloc_zeros(len)
                .map(Self::I64)
                .map_err(|error| cuda_error("zeros", error)),
            DType::BF16 => stream
                .alloc_zeros(len)
                .map(Self::BF16)
                .map_err(|error| cuda_error("zeros", error)),
            DType::F16 => stream
                .alloc_zeros(len)
                .map(Self::F16)
                .map_err(|error| cuda_error("zeros", error)),
            DType::F32 => stream
                .alloc_zeros(len)
                .map(Self::F32)
                .map_err(|error| cuda_error("zeros", error)),
            DType::F64 => stream
                .alloc_zeros(len)
                .map(Self::F64)
                .map_err(|error| cuda_error("zeros", error)),
        }
    }

    fn ones(stream: &Arc<CudaStream>, dtype: DType, len: usize) -> Result<Self> {
        macro_rules! ones {
            ($ty:ty, $variant:ident, $value:expr) => {
                stream
                    .clone_htod(&vec![$value as $ty; len])
                    .map(Self::$variant)
                    .map_err(|error| cuda_error("ones", error))
            };
        }

        match dtype {
            DType::U8 => ones!(u8, U8, 1),
            DType::U32 => ones!(u32, U32, 1),
            DType::I16 => ones!(i16, I16, 1),
            DType::I32 => ones!(i32, I32, 1),
            DType::I64 => ones!(i64, I64, 1),
            DType::BF16 => ones!(bf16, BF16, bf16::from_f32(1.0)),
            DType::F16 => ones!(f16, F16, f16::from_f32(1.0)),
            DType::F32 => ones!(f32, F32, 1.0),
            DType::F64 => ones!(f64, F64, 1.0),
        }
    }

    fn try_clone(&self) -> Result<Self> {
        macro_rules! clone_slice {
            ($data:expr, $variant:ident) => {
                $data
                    .try_clone()
                    .map(Self::$variant)
                    .map_err(|error| cuda_error("storage clone", error))
            };
        }

        match self {
            Self::U8(data) => clone_slice!(data, U8),
            Self::U32(data) => clone_slice!(data, U32),
            Self::I16(data) => clone_slice!(data, I16),
            Self::I32(data) => clone_slice!(data, I32),
            Self::I64(data) => clone_slice!(data, I64),
            Self::BF16(data) => clone_slice!(data, BF16),
            Self::F16(data) => clone_slice!(data, F16),
            Self::F32(data) => clone_slice!(data, F32),
            Self::F64(data) => clone_slice!(data, F64),
        }
    }

    pub(crate) fn base_ptr(&self, stream: &CudaStream) -> *const u8 {
        macro_rules! ptr {
            ($data:expr) => {{
                let (ptr, _sync) = $data.device_ptr(stream);
                ptr as *const u8
            }};
        }

        match self {
            Self::U8(data) => ptr!(data),
            Self::U32(data) => ptr!(data),
            Self::I16(data) => ptr!(data),
            Self::I32(data) => ptr!(data),
            Self::I64(data) => ptr!(data),
            Self::BF16(data) => ptr!(data),
            Self::F16(data) => ptr!(data),
            Self::F32(data) => ptr!(data),
            Self::F64(data) => ptr!(data),
        }
    }
}

/// A CUDA allocation together with the device that owns it.
#[derive(Debug)]
pub struct CudaStorage {
    pub(crate) data: CudaStorageSlice,
    device: Arc<CudaDevice>,
}

impl CudaStorage {
    pub(crate) fn from_data(device: Arc<CudaDevice>, data: CudaStorageSlice) -> Self {
        Self { data, device }
    }

    pub(crate) fn zeros(device: Arc<CudaDevice>, dtype: DType, len: usize) -> Result<Self> {
        let data = CudaStorageSlice::zeros(&device.cuda_stream(), dtype, len)?;
        Ok(Self { data, device })
    }

    pub(crate) fn ones(device: Arc<CudaDevice>, dtype: DType, len: usize) -> Result<Self> {
        let data = CudaStorageSlice::ones(&device.cuda_stream(), dtype, len)?;
        Ok(Self { data, device })
    }

    pub(crate) fn from_host_vec<T: DeviceRepr>(
        device: Arc<CudaDevice>,
        data: Vec<T>,
    ) -> Result<CudaSlice<T>> {
        device
            .cuda_stream()
            .clone_htod(&data)
            .map_err(|error| cuda_error("host to device copy", error))
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn device_handle(&self) -> Arc<CudaDevice> {
        Arc::clone(&self.device)
    }

    pub(crate) fn base_ptr(&self) -> *const u8 {
        self.data.base_ptr(self.device.cuda_stream().as_ref())
    }

    pub(crate) fn base_alignment(&self) -> usize {
        // CUDA device pointers are not host allocations. Do not promise a
        // host SIMD alignment to callers of the CPU-oriented metadata API.
        1
    }

    pub(crate) fn effective_alignment(&self, layout: &Layout) -> Result<usize> {
        crate::storage::validate_layout_for_storage(layout, self.len())?;
        Err(Error::UnsupportedCudaOp {
            op: "effective_alignment",
        })
    }

    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.try_clone()?,
            device: Arc::clone(&self.device),
        })
    }

    pub(crate) fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        Err(Error::UnsupportedCudaOp { op: "to_dtype" })
    }
}

impl BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn device(&self) -> &Self::Device {
        self.device()
    }

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        self.try_clone()
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        self.to_dtype(layout, dtype)
    }
}

pub(crate) fn cuda_error(op: &'static str, error: impl Display) -> Error {
    Error::CudaOperationFailed {
        op,
        message: error.to_string(),
    }
}
