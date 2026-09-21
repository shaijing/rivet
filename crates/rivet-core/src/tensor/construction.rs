use super::{Tensor, Tensor_};
use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::storage::{Storage, validate_layout_for_storage};
use crate::{DType, Device, Error, Layout, Result, Shape, WithDType};
use std::ops::Add;
use std::sync::Arc;

/// Numeric element contract used by [`Tensor::arange`] and
/// [`Tensor::arange_step`].
///
/// This is deliberately separate from `WithDType`: dtype conversion and
/// storage access do not require ordering or arithmetic semantics.
pub trait RangeElement: WithDType + PartialEq + PartialOrd + Add<Output = Self> {
    fn zero() -> Self;
    fn one() -> Self;
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_range_element_int {
    ($ty:ty) => {
        impl RangeElement for $ty {
            fn zero() -> Self {
                0
            }

            fn one() -> Self {
                1
            }

            fn checked_add(self, rhs: Self) -> Option<Self> {
                self.checked_add(rhs)
            }
        }
    };
}

macro_rules! impl_range_element_float {
    ($ty:ty) => {
        impl RangeElement for $ty {
            fn zero() -> Self {
                0.0
            }

            fn one() -> Self {
                1.0
            }

            fn checked_add(self, rhs: Self) -> Option<Self> {
                let next = self + rhs;
                next.is_finite().then_some(next)
            }
        }
    };
}

impl_range_element_int!(u8);
impl_range_element_int!(u32);
impl_range_element_int!(i16);
impl_range_element_int!(i32);
impl_range_element_int!(i64);
impl_range_element_float!(f32);
impl_range_element_float!(f64);

impl RangeElement for half::bf16 {
    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    fn one() -> Self {
        Self::from_f32(1.0)
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        let next = self + rhs;
        next.is_finite().then_some(next)
    }
}

impl RangeElement for half::f16 {
    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    fn one() -> Self {
        Self::from_f32(1.0)
    }

    fn checked_add(self, rhs: Self) -> Option<Self> {
        let next = self + rhs;
        next.is_finite().then_some(next)
    }
}

impl Tensor {
    pub(super) fn from_parts(
        storage: Arc<Storage>,
        layout: Layout,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        {
            if storage.dtype() != dtype || !storage.device().same_device(&device) {
                return Err(if storage.dtype() != dtype {
                    Error::UnexpectedDType {
                        expected: dtype,
                        actual: storage.dtype(),
                    }
                } else {
                    Error::DeviceMismatch
                });
            }
            validate_layout_for_storage(&layout, storage.len())?;
        }
        Ok(Self(Arc::new(Tensor_ {
            id: super::TensorId::new(),
            storage,
            layout,
            dtype,
            device,
        })))
    }

    /// Creates a view from a layout derived from this tensor's already-valid
    /// layout. Callers must only pass layouts produced by checked view
    /// transformations such as narrow, permute, or reshape.
    pub(super) fn from_validated_shared_storage(&self, layout: Layout) -> Self {
        Self(Arc::new(Tensor_ {
            id: super::TensorId::new(),
            storage: Arc::clone(&self.0.storage),
            layout,
            dtype: self.0.dtype,
            device: self.0.device.clone(),
        }))
    }

    /// Creates a tensor from an owned, exactly-sized storage allocation.
    pub fn from_storage(
        storage: Storage,
        shape: impl Into<Shape>,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.elem_count();
        let actual = storage.len();
        if expected != actual {
            return Err(Error::ShapeMismatch { expected, actual });
        }
        if !storage.device().same_device(device) {
            return Err(Error::DeviceMismatch);
        }
        let dtype = storage.dtype();
        Self::from_parts(
            Arc::new(storage),
            Layout::contiguous(shape),
            dtype,
            device.clone(),
        )
    }

    pub fn from_vec<T, S>(data: Vec<T>, shape: S, device: &Device) -> Result<Self>
    where
        T: WithDType,
        S: Into<Shape>,
    {
        let storage = match device {
            Device::Cpu => CpuDevice.storage_from_vec(data)?,
        };
        Self::from_storage(Storage::Cpu(storage), shape, device)
    }

    pub fn from_slice<T, S>(data: &[T], shape: S, device: &Device) -> Result<Self>
    where
        T: WithDType,
        S: Into<Shape>,
    {
        let storage = match device {
            Device::Cpu => CpuDevice.storage_from_slice(data)?,
        };
        Self::from_storage(Storage::Cpu(storage), shape, device)
    }

    pub fn zeros<S>(shape: S, dtype: DType, device: &Device) -> Result<Self>
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let storage = match device {
            Device::Cpu => CpuDevice.zeros(&shape, dtype)?,
        };
        Self::from_storage(Storage::Cpu(storage), shape, device)
    }

    pub fn ones<S>(shape: S, dtype: DType, device: &Device) -> Result<Self>
    where
        S: Into<Shape>,
    {
        let shape = shape.into();
        let storage = match device {
            Device::Cpu => CpuDevice.ones(&shape, dtype)?,
        };
        Self::from_storage(Storage::Cpu(storage), shape, device)
    }

    /// Creates a tensor filled with one scalar value.
    pub fn full<D, S>(value: D, shape: S, device: &Device) -> Result<Self>
    where
        D: WithDType,
        S: Into<Shape>,
    {
        let shape = shape.into();
        Self::from_vec(vec![value; shape.elem_count()], shape, device)
    }

    /// Creates a one-dimensional tensor from an iterator.
    pub fn from_iter<D>(iter: impl IntoIterator<Item = D>, device: &Device) -> Result<Self>
    where
        D: WithDType,
    {
        let values = iter.into_iter().collect::<Vec<_>>();
        let len = values.len();
        Self::from_vec(values, len, device)
    }

    /// Creates values in the half-open interval `[start, end)` with step one.
    pub fn arange<D>(start: D, end: D, device: &Device) -> Result<Self>
    where
        D: RangeElement,
    {
        Self::arange_step(start, end, D::one(), device)
    }

    /// Creates values in a half-open range using an explicit step.
    pub fn arange_step<D>(start: D, end: D, step: D, device: &Device) -> Result<Self>
    where
        D: RangeElement,
    {
        if step == D::zero() {
            return Err(Error::InvalidRangeStep);
        }

        let increasing = step > D::zero();
        let mut current = start;
        let mut values = Vec::new();
        while if increasing {
            current < end
        } else {
            current > end
        } {
            values.push(current);
            let next = current.checked_add(step).ok_or(Error::RangeOverflow)?;
            if (increasing && next <= current) || (!increasing && next >= current) {
                return Err(Error::RangeOverflow);
            }
            current = next;
        }

        let len = values.len();
        Self::from_vec(values, len, device)
    }

    /// Creates a tensor filled with zeros and matching this tensor's shape,
    /// dtype, and device.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape().clone(), self.dtype(), self.device())
    }

    /// Creates a tensor filled with ones and matching this tensor's shape,
    /// dtype, and device.
    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape().clone(), self.dtype(), self.device())
    }
}
