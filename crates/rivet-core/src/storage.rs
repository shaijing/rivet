use crate::cpu_backend::CpuStorage;
use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::{DType, Device, Error, Layout, Result, Shape, WithDType};
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

/// Backend storage. Storage itself is deliberately not `Clone`; cloning an
/// allocation is explicit through `try_clone`.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
}

pub type StorageRef<'a> = RwLockReadGuard<'a, Storage>;
pub type StorageMutRef<'a> = RwLockWriteGuard<'a, Storage>;

impl Storage {
    pub(crate) fn binary(
        lhs: &Self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Self> {
        match (lhs, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                Ok(Self::Cpu(lhs.binary(lhs_layout, rhs, rhs_layout, op)?))
            }
        }
    }

    pub(crate) fn binary_scalar<T: WithDType>(
        storage: &Self,
        layout: &Layout,
        scalar: T,
        op: BinaryOp,
    ) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.binary_scalar(layout, scalar, op)?)),
        }
    }

    pub(crate) fn unary(storage: &Self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.unary(layout, op)?)),
        }
    }

    pub(crate) fn cmp(
        lhs: &Self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: CmpOp,
    ) -> Result<Self> {
        match (lhs, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                Ok(Self::Cpu(lhs.cmp(lhs_layout, rhs, rhs_layout, op)?))
            }
        }
    }

    pub(crate) fn cmp_scalar<T: WithDType>(
        storage: &Self,
        layout: &Layout,
        scalar: T,
        op: CmpOp,
    ) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.cmp_scalar(layout, scalar, op)?)),
        }
    }

    pub(crate) fn where_cond(
        condition: &Self,
        condition_layout: &Layout,
        on_true: &Self,
        true_layout: &Layout,
        on_false: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        match (condition, on_true, on_false) {
            (Self::Cpu(condition), Self::Cpu(on_true), Self::Cpu(on_false)) => {
                Ok(Self::Cpu(CpuStorage::where_cond(
                    condition,
                    condition_layout,
                    on_true,
                    true_layout,
                    on_false,
                    false_layout,
                )?))
            }
        }
    }

    pub(crate) fn reduce_dim(
        storage: &Self,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
        op: ReduceOp,
    ) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.reduce_dim(layout, dim, keepdim, op)?)),
        }
    }

    pub(crate) fn reduce_all(storage: &Self, layout: &Layout, op: ReduceOp) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.reduce_all(layout, op)?)),
        }
    }

    pub(crate) fn mean_dim(
        storage: &Self,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
    ) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.mean_dim(layout, dim, keepdim)?)),
        }
    }

    pub(crate) fn mean_all(storage: &Self, layout: &Layout) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.mean_all(layout)?)),
        }
    }

    pub(crate) fn var_dim(
        storage: &Self,
        layout: &Layout,
        dim: usize,
        keepdim: bool,
    ) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.var_dim(layout, dim, keepdim)?)),
        }
    }

    pub(crate) fn copy_logical(storage: &Self, layout: &Layout) -> Result<Self> {
        match storage {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.copy_logical(layout)?)),
        }
    }

    pub(crate) fn cat(
        inputs: &[(&Self, &Layout)],
        output_shape: &Shape,
        dim: usize,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::EmptyTensorList);
        }
        let mut cpu_inputs = Vec::with_capacity(inputs.len());
        for (storage, layout) in inputs {
            match storage {
                Self::Cpu(storage) => cpu_inputs.push((storage, *layout)),
            }
        }
        Ok(Self::Cpu(CpuStorage::cat(&cpu_inputs, output_shape, dim)?))
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
        }
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        self.device().same_device(&rhs.device())
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(storage) => storage.len(),
        }
    }

    pub fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
        }
    }

    /// Convert/materialize a logical layout into contiguous storage.
    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        validate_layout_for_storage(layout, self.len())?;
        if dtype == self.dtype() {
            return Self::copy_logical(self, layout);
        }
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.to_dtype(layout, dtype)?)),
        }
    }
}

/// Validates that every logical element address is representable by a storage
/// allocation. Empty layouts do not address any element and are therefore
/// valid even when their offset is at the end of the allocation.
pub(crate) fn validate_layout_for_storage(layout: &Layout, storage_len: usize) -> Result<()> {
    if layout.elem_count() == 0 {
        return Ok(());
    }

    let mut max_offset = layout.start_offset();
    for (&dim, &stride) in layout.dims().iter().zip(layout.stride()) {
        if dim > 0 {
            let span = (dim - 1)
                .checked_mul(stride)
                .ok_or(Error::StorageOutOfBounds)?;
            max_offset = max_offset
                .checked_add(span)
                .ok_or(Error::StorageOutOfBounds)?;
        }
    }
    if max_offset >= storage_len {
        return Err(Error::StorageOutOfBounds);
    }
    Ok(())
}
