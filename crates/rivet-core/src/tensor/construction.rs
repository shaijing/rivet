use super::{Tensor, Tensor_};
use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::storage::{Storage, validate_layout_for_storage};
use crate::{DType, Device, Error, Layout, Result, Shape, WithDType};
use std::sync::{Arc, RwLock};

impl Tensor {
    pub(super) fn from_parts(
        storage: Arc<RwLock<Storage>>,
        layout: Layout,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        {
            let storage_guard = storage.read().expect("tensor storage lock poisoned");
            if storage_guard.dtype() != dtype || !storage_guard.device().same_device(&device) {
                return Err(if storage_guard.dtype() != dtype {
                    Error::UnexpectedDType {
                        expected: dtype,
                        actual: storage_guard.dtype(),
                    }
                } else {
                    Error::DeviceMismatch
                });
            }
            validate_layout_for_storage(&layout, storage_guard.len())?;
        }
        Ok(Self(Arc::new(Tensor_ {
            id: super::TensorId::new(),
            storage,
            layout,
            dtype,
            device,
        })))
    }

    pub(super) fn from_shared_storage(&self, layout: Layout) -> Result<Self> {
        Self::from_parts(
            Arc::clone(&self.0.storage),
            layout,
            self.0.dtype,
            self.0.device.clone(),
        )
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
            Arc::new(RwLock::new(storage)),
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
}
