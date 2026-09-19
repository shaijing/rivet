use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::ops::{BinaryOp, UnaryOp};
use crate::storage::{Storage, StorageMutRef, StorageRef, validate_layout_for_storage};
use crate::{DType, Device, Error, Layout, Result, Shape, WithDType};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Unique identifier for a logical tensor node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    fn new() -> Self {
        static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
struct Tensor_ {
    id: TensorId,
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    dtype: DType,
    device: Device,
}

/// A cheap, reference-counted tensor handle.
#[derive(Debug, Clone)]
pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    fn from_parts(
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
            id: TensorId::new(),
            storage,
            layout,
            dtype,
            device,
        })))
    }

    fn from_shared_storage(&self, layout: Layout) -> Result<Self> {
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

    pub fn id(&self) -> TensorId {
        self.0.id
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.layout().dims()
    }

    pub fn stride(&self) -> &[usize] {
        self.layout().stride()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    /// Number of bytes in the logical tensor represented by this handle.
    pub fn logical_bytes(&self) -> usize {
        self.elem_count()
            .saturating_mul(self.dtype().size_in_bytes())
    }

    /// Number of bytes in the backing allocation, including bytes outside a
    /// view's logical region.
    pub fn storage_bytes(&self) -> usize {
        self.storage()
            .len()
            .saturating_mul(self.dtype().size_in_bytes())
    }

    /// Returns whether two tensor handles refer to the same backing storage.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0.storage, &other.0.storage)
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        self.layout().is_fortran_contiguous()
    }

    pub fn strided_index(&self) -> crate::StridedIndex<'_> {
        self.layout().strided_index()
    }

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

    /// Extracts values in logical row-major order, independent of the view's
    /// physical storage order.
    pub fn to_vec<T: WithDType>(&self) -> Result<Vec<T>> {
        let storage = self.storage();
        let cpu_storage = match &*storage {
            Storage::Cpu(storage) => storage,
        };
        let values = T::cpu_storage_as_slice(cpu_storage)?;
        if let Some((start, end)) = self.layout().contiguous_offsets() {
            return values
                .get(start..end)
                .map(ToOwned::to_owned)
                .ok_or(Error::StorageOutOfBounds);
        }
        self.layout()
            .strided_index()
            .map(|index| values.get(index).copied().ok_or(Error::StorageOutOfBounds))
            .collect()
    }

    /// Visits values in logical row-major order without allocating a second
    /// vector for a view. The callback only receives copied values, so it
    /// cannot mutate or retain a reference to tensor storage.
    pub fn for_each<T, F>(&self, mut f: F) -> Result<()>
    where
        T: WithDType,
        F: FnMut(T),
    {
        let storage = self.storage();
        let cpu_storage = match &*storage {
            Storage::Cpu(storage) => storage,
        };
        let values = T::cpu_storage_as_slice(cpu_storage)?;
        if let Some((start, end)) = self.layout().contiguous_offsets() {
            for &value in values.get(start..end).ok_or(Error::StorageOutOfBounds)? {
                f(value);
            }
        } else {
            for index in self.layout().strided_index() {
                f(*values.get(index).ok_or(Error::StorageOutOfBounds)?);
            }
        }
        Ok(())
    }

    pub fn to_vec0<T: WithDType>(&self) -> Result<T> {
        if self.rank() != 0 {
            return Err(Error::InvalidRank {
                expected: 0,
                actual: self.rank(),
            });
        }
        self.to_vec()?
            .into_iter()
            .next()
            .ok_or(Error::StorageOutOfBounds)
    }

    pub fn to_vec1<T: WithDType>(&self) -> Result<Vec<T>> {
        if self.rank() != 1 {
            return Err(Error::InvalidRank {
                expected: 1,
                actual: self.rank(),
            });
        }
        self.to_vec()
    }

    pub fn to_vec2<T: WithDType>(&self) -> Result<Vec<Vec<T>>> {
        if self.rank() != 2 {
            return Err(Error::InvalidRank {
                expected: 2,
                actual: self.rank(),
            });
        }
        let values = self.to_vec::<T>()?;
        let width = self.dims()[1];
        if width == 0 {
            return Ok((0..self.dims()[0]).map(|_| Vec::new()).collect());
        }
        Ok(values.chunks(width).map(ToOwned::to_owned).collect())
    }

    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    fn binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self> {
        if self.shape() != rhs.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }

        let lhs_storage = self.storage();
        let rhs_storage = rhs.storage();
        let storage = Storage::binary(&lhs_storage, self.layout(), &rhs_storage, rhs.layout(), op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    fn broadcast_binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        let shape = self.shape().broadcast_shape_binary_op(rhs.shape())?;
        let lhs = self.broadcast_as(shape.clone())?;
        let rhs = rhs.broadcast_as(shape)?;
        lhs.binary(&rhs, op)
    }

    fn unary(&self, op: UnaryOp) -> Result<Self> {
        let storage = self.storage();
        let storage = Storage::unary(&storage, self.layout(), op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    fn binary_scalar<T: WithDType>(&self, scalar: T, op: BinaryOp) -> Result<Self> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: T::DTYPE,
            });
        }
        let storage = self.storage();
        let storage = Storage::binary_scalar(&storage, self.layout(), scalar, op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Add)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Sub)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Mul)
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Div)
    }

    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Add)
    }

    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Sub)
    }

    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Mul)
    }

    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Div)
    }

    pub fn add_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Add)
    }

    pub fn sub_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Sub)
    }

    pub fn mul_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Mul)
    }

    pub fn div_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Div)
    }

    pub fn neg(&self) -> Result<Self> {
        self.unary(UnaryOp::Neg)
    }

    pub fn abs(&self) -> Result<Self> {
        self.unary(UnaryOp::Abs)
    }

    pub fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        let first = tensors.first().ok_or(Error::EmptyTensorList)?;
        let rank = first.rank();
        if dim >= rank {
            return Err(Error::InvalidConcatDim { dim, rank });
        }

        let mut output_dims = first.dims().to_vec();
        output_dims[dim] = 0;
        for tensor in tensors {
            if tensor.rank() != rank {
                return Err(Error::ShapeMismatchBinary {
                    lhs: first.dims().to_vec(),
                    rhs: tensor.dims().to_vec(),
                });
            }
            if tensor.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch {
                    lhs: first.dtype(),
                    rhs: tensor.dtype(),
                });
            }
            if !tensor.device().same_device(first.device()) {
                return Err(Error::DeviceMismatch);
            }
            for axis in 0..rank {
                if axis != dim && tensor.dims()[axis] != first.dims()[axis] {
                    return Err(Error::ShapeMismatchBinary {
                        lhs: first.dims().to_vec(),
                        rhs: tensor.dims().to_vec(),
                    });
                }
            }
            output_dims[dim] = output_dims[dim]
                .checked_add(tensor.dims()[dim])
                .ok_or(Error::StorageOutOfBounds)?;
        }

        let guards: Vec<_> = tensors.iter().map(|tensor| tensor.storage()).collect();
        let inputs: Vec<_> = guards
            .iter()
            .zip(tensors.iter())
            .map(|(storage, tensor)| (&**storage, tensor.layout()))
            .collect();
        let output_shape = Shape::from(output_dims);
        let storage = Storage::cat(&inputs, &output_shape, dim)?;
        Self::from_storage(storage, output_shape, first.device())
    }

    pub fn stack(tensors: &[&Self], dim: usize) -> Result<Self> {
        let first = tensors.first().ok_or(Error::EmptyTensorList)?;
        if dim > first.rank() {
            return Err(Error::InvalidConcatDim {
                dim,
                rank: first.rank() + 1,
            });
        }
        let expanded: Vec<Self> = tensors
            .iter()
            .map(|tensor| tensor.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        let expanded_refs: Vec<&Self> = expanded.iter().collect();
        Self::cat(&expanded_refs, dim)
    }

    fn storage(&self) -> StorageRef<'_> {
        self.0.storage.read().expect("tensor storage lock poisoned")
    }

    #[allow(dead_code)]
    fn storage_mut(&self) -> StorageMutRef<'_> {
        self.0
            .storage
            .write()
            .expect("tensor storage lock poisoned")
    }
}
