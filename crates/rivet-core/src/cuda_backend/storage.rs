use std::fmt::Display;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, DeviceRepr};
use half::{bf16, f16};

use crate::backend::BackendStorage;
use crate::cpu_backend::{CpuStorage, CpuStorageRef};
use crate::ops::{BinaryOp, CmpOp, UnaryOp};
use crate::{DType, Error, Layout, Result, WithDType};

use super::device::CudaDevice;
use super::kernels;
use super::matmul;

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

/// A typed view into one CUDA allocation. The view only carries the base
/// offset and length; tensor shape and stride remain owned by [`Layout`].
#[derive(Debug)]
pub enum CudaStorageView<'a> {
    U8(CudaView<'a, u8>),
    U32(CudaView<'a, u32>),
    I16(CudaView<'a, i16>),
    I32(CudaView<'a, i32>),
    I64(CudaView<'a, i64>),
    BF16(CudaView<'a, bf16>),
    F16(CudaView<'a, f16>),
    F32(CudaView<'a, f32>),
    F64(CudaView<'a, f64>),
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

    /// Returns a typed device view over `start..start + len` elements.
    /// Tensor shape and stride metadata must remain with the caller's layout.
    pub fn view(&self, start: usize, len: usize) -> Result<CudaStorageView<'_>> {
        let end = start.checked_add(len).ok_or(Error::StorageOutOfBounds)?;
        macro_rules! view {
            ($data:expr, $variant:ident) => {
                $data
                    .try_slice(start..end)
                    .map(CudaStorageView::$variant)
                    .ok_or(Error::StorageOutOfBounds)
            };
        }

        match self {
            Self::U8(data) => view!(data, U8),
            Self::U32(data) => view!(data, U32),
            Self::I16(data) => view!(data, I16),
            Self::I32(data) => view!(data, I32),
            Self::I64(data) => view!(data, I64),
            Self::BF16(data) => view!(data, BF16),
            Self::F16(data) => view!(data, F16),
            Self::F32(data) => view!(data, F32),
            Self::F64(data) => view!(data, F64),
        }
    }

    pub(crate) fn zeros(stream: &Arc<CudaStream>, dtype: DType, len: usize) -> Result<Self> {
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

    fn ones(device: &CudaDevice, dtype: DType, len: usize) -> Result<Self> {
        let mut output = Self::zeros(&device.cuda_stream(), dtype, len)?;
        kernels::fill_one(device, &mut output)?;
        Ok(output)
    }

    fn try_clone(&self, device: &CudaDevice) -> Result<Self> {
        if !self.is_empty() {
            device.record_d2d();
        }
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

    fn clone_range(&self, device: &CudaDevice, start: usize, len: usize) -> Result<Self> {
        let stream = device.cuda_stream();
        if len > 0 {
            device.record_d2d();
        }
        // The enum view cannot be projected with a common method, so dispatch
        // once here and keep the actual copy typed.
        match self.view(start, len)? {
            CudaStorageView::U8(view) => stream
                .clone_dtod(&view)
                .map(Self::U8)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::U32(view) => stream
                .clone_dtod(&view)
                .map(Self::U32)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::I16(view) => stream
                .clone_dtod(&view)
                .map(Self::I16)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::I32(view) => stream
                .clone_dtod(&view)
                .map(Self::I32)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::I64(view) => stream
                .clone_dtod(&view)
                .map(Self::I64)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::BF16(view) => stream
                .clone_dtod(&view)
                .map(Self::BF16)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::F16(view) => stream
                .clone_dtod(&view)
                .map(Self::F16)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::F32(view) => stream
                .clone_dtod(&view)
                .map(Self::F32)
                .map_err(|error| cuda_error("device to device copy", error)),
            CudaStorageView::F64(view) => stream
                .clone_dtod(&view)
                .map(Self::F64)
                .map_err(|error| cuda_error("device to device copy", error)),
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
        let data = CudaStorageSlice::ones(device.as_ref(), dtype, len)?;
        Ok(Self { data, device })
    }

    pub(crate) fn from_host_vec<T: DeviceRepr>(
        device: Arc<CudaDevice>,
        data: Vec<T>,
    ) -> Result<CudaSlice<T>> {
        Self::from_host_slice(device, &data)
    }

    pub(crate) fn from_host_slice<T: DeviceRepr>(
        device: Arc<CudaDevice>,
        data: &[T],
    ) -> Result<CudaSlice<T>> {
        if !data.is_empty() {
            device.record_h2d();
        }
        device
            .cuda_stream()
            .clone_htod(data)
            .map_err(|error| cuda_error("host to device copy", error))
    }

    pub(crate) fn from_cpu_storage(
        device: Arc<CudaDevice>,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<Self> {
        crate::storage::validate_layout_for_storage(layout, storage.len())?;

        macro_rules! copy_host {
            ($variant:ident, $values:expr) => {{
                let allocation = copy_host_values(&device, $values, layout)?;
                Ok(Self::from_data(
                    Arc::clone(&device),
                    CudaStorageSlice::$variant(allocation),
                ))
            }};
        }

        match storage.as_ref() {
            CpuStorageRef::U8(values) => copy_host!(U8, values),
            CpuStorageRef::U32(values) => copy_host!(U32, values),
            CpuStorageRef::I16(values) => copy_host!(I16, values),
            CpuStorageRef::I32(values) => copy_host!(I32, values),
            CpuStorageRef::I64(values) => copy_host!(I64, values),
            CpuStorageRef::BF16(values) => copy_host!(BF16, values),
            CpuStorageRef::F16(values) => copy_host!(F16, values),
            CpuStorageRef::F32(values) => copy_host!(F32, values),
            CpuStorageRef::F64(values) => copy_host!(F64, values),
        }
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
            data: self.data.try_clone(self.device.as_ref())?,
            device: Arc::clone(&self.device),
        })
    }

    pub(crate) fn to_cpu_storage(&self, layout: &Layout) -> Result<CpuStorage> {
        crate::storage::validate_layout_for_storage(layout, self.len())?;

        macro_rules! copy_device {
            ($variant:ident, $ty:ty, $data:expr) => {{
                let values = copy_device_values($data, layout, self.device.as_ref())?;
                <$ty as WithDType>::into_cpu_storage(values)
            }};
        }

        match &self.data {
            CudaStorageSlice::U8(data) => copy_device!(U8, u8, data),
            CudaStorageSlice::U32(data) => copy_device!(U32, u32, data),
            CudaStorageSlice::I16(data) => copy_device!(I16, i16, data),
            CudaStorageSlice::I32(data) => copy_device!(I32, i32, data),
            CudaStorageSlice::I64(data) => copy_device!(I64, i64, data),
            CudaStorageSlice::BF16(data) => copy_device!(BF16, bf16, data),
            CudaStorageSlice::F16(data) => copy_device!(F16, f16, data),
            CudaStorageSlice::F32(data) => copy_device!(F32, f32, data),
            CudaStorageSlice::F64(data) => copy_device!(F64, f64, data),
        }
    }

    pub(crate) fn copy_logical(&self, layout: &Layout) -> Result<Self> {
        self.copy_to_device(Arc::clone(&self.device), layout)
    }

    pub(crate) fn matmul(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if !self.device.same_device(&rhs.device) {
            return Err(Error::DeviceMismatch);
        }
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !matches!(self.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedMatmulDType {
                dtype: self.dtype(),
            });
        }
        crate::storage::validate_layout_for_storage(lhs_layout, self.len())?;
        crate::storage::validate_layout_for_storage(rhs_layout, rhs.len())?;

        let plan = matmul::MatmulPlan::new(lhs_layout, rhs_layout)?;
        let mut output =
            CudaStorageSlice::zeros(&self.device.cuda_stream(), self.dtype(), plan.output_len()?)?;
        if plan.has_empty_dimension() {
            return Ok(Self::from_data(Arc::clone(&self.device), output));
        }

        let lhs_layout_is_general = plan.needs_lhs_materialization();
        let rhs_layout_is_general = plan.needs_rhs_materialization();
        match (lhs_layout_is_general, rhs_layout_is_general) {
            (false, false) => matmul::matmul(
                self.device.as_ref(),
                &self.data,
                lhs_layout,
                &rhs.data,
                rhs_layout,
                &plan,
                &mut output,
            )?,
            (true, false) => {
                let lhs_materialized = self.copy_to_device(Arc::clone(&self.device), lhs_layout)?;
                let lhs_layout = Layout::contiguous(lhs_layout.dims().to_vec());
                let plan = matmul::MatmulPlan::new(&lhs_layout, rhs_layout)?;
                matmul::matmul(
                    self.device.as_ref(),
                    &lhs_materialized.data,
                    &lhs_layout,
                    &rhs.data,
                    rhs_layout,
                    &plan,
                    &mut output,
                )?;
            }
            (false, true) => {
                let rhs_materialized = rhs.copy_to_device(Arc::clone(&self.device), rhs_layout)?;
                let rhs_layout = Layout::contiguous(rhs_layout.dims().to_vec());
                let plan = matmul::MatmulPlan::new(lhs_layout, &rhs_layout)?;
                matmul::matmul(
                    self.device.as_ref(),
                    &self.data,
                    lhs_layout,
                    &rhs_materialized.data,
                    &rhs_layout,
                    &plan,
                    &mut output,
                )?;
            }
            (true, true) => {
                let lhs_materialized = self.copy_to_device(Arc::clone(&self.device), lhs_layout)?;
                let rhs_materialized = rhs.copy_to_device(Arc::clone(&self.device), rhs_layout)?;
                let lhs_layout = Layout::contiguous(lhs_layout.dims().to_vec());
                let rhs_layout = Layout::contiguous(rhs_layout.dims().to_vec());
                let plan = matmul::MatmulPlan::new(&lhs_layout, &rhs_layout)?;
                matmul::matmul(
                    self.device.as_ref(),
                    &lhs_materialized.data,
                    &lhs_layout,
                    &rhs_materialized.data,
                    &rhs_layout,
                    &plan,
                    &mut output,
                )?;
            }
        }

        Ok(Self::from_data(Arc::clone(&self.device), output))
    }

    pub(crate) fn unary(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        let numel = layout.checked_elem_count()?;
        let mut output = CudaStorageSlice::zeros(&self.device.cuda_stream(), self.dtype(), numel)?;
        kernels::unary(self.device.as_ref(), &self.data, &mut output, layout, op)?;
        Ok(Self::from_data(Arc::clone(&self.device), output))
    }

    pub(crate) fn binary(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Self> {
        if !self.device.same_device(&rhs.device) {
            return Err(Error::DeviceMismatch);
        }
        let numel = lhs_layout.checked_elem_count()?;
        if numel != rhs_layout.checked_elem_count()? {
            return Err(Error::ShapeMismatchBinary {
                lhs: lhs_layout.dims().to_vec(),
                rhs: rhs_layout.dims().to_vec(),
            });
        }
        let mut output = CudaStorageSlice::zeros(&self.device.cuda_stream(), self.dtype(), numel)?;
        kernels::binary(
            self.device.as_ref(),
            &self.data,
            lhs_layout,
            &rhs.data,
            rhs_layout,
            &mut output,
            op,
        )?;
        Ok(Self::from_data(Arc::clone(&self.device), output))
    }

    pub(crate) fn binary_scalar<T: WithDType>(
        &self,
        layout: &Layout,
        scalar: T,
        op: BinaryOp,
    ) -> Result<Self> {
        let numel = layout.checked_elem_count()?;
        let mut output = CudaStorageSlice::zeros(&self.device.cuda_stream(), self.dtype(), numel)?;
        kernels::binary_scalar(
            self.device.as_ref(),
            &self.data,
            layout,
            &mut output,
            scalar,
            op,
        )?;
        Ok(Self::from_data(Arc::clone(&self.device), output))
    }

    pub(crate) fn cmp(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: CmpOp,
    ) -> Result<Self> {
        if !self.device.same_device(&rhs.device) {
            return Err(Error::DeviceMismatch);
        }
        let numel = lhs_layout.checked_elem_count()?;
        if numel != rhs_layout.checked_elem_count()? {
            return Err(Error::ShapeMismatchBinary {
                lhs: lhs_layout.dims().to_vec(),
                rhs: rhs_layout.dims().to_vec(),
            });
        }
        let mut output = CudaStorageSlice::zeros(&self.device.cuda_stream(), DType::U8, numel)?;
        kernels::compare(
            self.device.as_ref(),
            &self.data,
            lhs_layout,
            &rhs.data,
            rhs_layout,
            &mut output,
            op,
        )?;
        Ok(Self::from_data(Arc::clone(&self.device), output))
    }

    pub(crate) fn cmp_scalar<T: WithDType>(
        &self,
        layout: &Layout,
        scalar: T,
        op: CmpOp,
    ) -> Result<Self> {
        let numel = layout.checked_elem_count()?;
        let mut output = CudaStorageSlice::zeros(&self.device.cuda_stream(), DType::U8, numel)?;
        kernels::compare_scalar(
            self.device.as_ref(),
            &self.data,
            layout,
            &mut output,
            scalar,
            op,
        )?;
        Ok(Self::from_data(Arc::clone(&self.device), output))
    }

    pub(crate) fn copy_to_device(&self, device: Arc<CudaDevice>, layout: &Layout) -> Result<Self> {
        crate::storage::validate_layout_for_storage(layout, self.len())?;
        if !self.device.same_device(&device) {
            return Err(Error::UnsupportedCudaOp {
                op: "cuda_device_copy",
            });
        }

        if layout.checked_elem_count()? == 0 {
            let data = self.data.clone_range(device.as_ref(), 0, 0)?;
            return Ok(Self::from_data(device, data));
        }

        if let Some((start, end)) = layout.contiguous_offsets() {
            let data = self.data.clone_range(device.as_ref(), start, end - start)?;
            return Ok(Self::from_data(device, data));
        }

        let numel = layout.checked_elem_count()?;
        let mut data = CudaStorageSlice::zeros(&device.cuda_stream(), self.dtype(), numel)?;
        kernels::copy_layout(device.as_ref(), &self.data, &mut data, layout)?;
        Ok(Self::from_data(device, data))
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

fn copy_host_values<T: DeviceRepr + Copy>(
    device: &Arc<CudaDevice>,
    values: &[T],
    layout: &Layout,
) -> Result<CudaSlice<T>> {
    if layout.checked_elem_count()? == 0 {
        return CudaStorage::from_host_slice(Arc::clone(device), &[]);
    }

    if let Some((start, end)) = layout.contiguous_offsets() {
        let values = values.get(start..end).ok_or(Error::StorageOutOfBounds)?;
        return CudaStorage::from_host_slice(Arc::clone(device), values);
    }

    CudaStorage::from_host_vec(Arc::clone(device), logical_values(values, layout)?)
}

fn copy_device_values<T: DeviceRepr + Copy>(
    data: &CudaSlice<T>,
    layout: &Layout,
    device: &CudaDevice,
) -> Result<Vec<T>> {
    if layout.checked_elem_count()? == 0 {
        return Ok(Vec::new());
    }

    if let Some((start, end)) = layout.contiguous_offsets() {
        let view = data
            .try_slice(start..end)
            .ok_or(Error::StorageOutOfBounds)?;
        let values = device
            .cuda_stream()
            .clone_dtoh(&view)
            .map_err(|error| cuda_error("device to host copy", error));
        let values = values?;
        device.record_d2h();
        device.synchronize()?;
        return Ok(values);
    }

    let values = device
        .cuda_stream()
        .clone_dtoh(data)
        .map_err(|error| cuda_error("device to host copy", error))?;
    device.record_d2h();
    device.synchronize()?;
    logical_values(&values, layout)
}

fn logical_values<T: Copy>(values: &[T], layout: &Layout) -> Result<Vec<T>> {
    if let Some((start, end)) = layout.contiguous_offsets() {
        return values
            .get(start..end)
            .map(ToOwned::to_owned)
            .ok_or(Error::StorageOutOfBounds);
    }

    layout
        .strided_index()
        .map(|index| values.get(index).copied().ok_or(Error::StorageOutOfBounds))
        .collect()
}
