use super::{ExclusiveTensor, Tensor, Tensor_, TensorId};
use crate::cpu_backend::CpuStorageRef;
use crate::storage::Storage;
use crate::{DType, Error, Layout, Result, WithDType};
use std::sync::Arc;

impl Tensor {
    /// Returns whether no other `Tensor` handle points at this logical tensor
    /// node.
    pub fn is_handle_unique(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }

    /// Returns whether no other tensor node points at the backing storage.
    ///
    /// This does not rule out `Tensor::clone()` aliases of this same handle;
    /// use [`Self::is_uniquely_owned`] for the complete ownership predicate.
    pub fn is_storage_unique(&self) -> bool {
        Arc::strong_count(&self.0.storage) == 1
    }

    /// Returns whether this handle and its backing storage are both unique.
    ///
    /// This is an observational check for diagnostics and fast paths. An
    /// ownership transfer must use [`Self::try_into_exclusive`] instead of
    /// relying on this result across a later operation.
    pub fn is_uniquely_owned(&self) -> bool {
        self.is_handle_unique() && self.is_storage_unique()
    }

    /// Atomically takes exclusive ownership of this tensor's backing storage.
    ///
    /// The operation uses `Arc::try_unwrap` for both reference-counted layers,
    /// so a concurrent clone or a view alias cannot pass a check-then-transfer
    /// race. Read-only or otherwise non-transferable storage is also rejected.
    pub fn try_into_exclusive(self) -> std::result::Result<ExclusiveTensor, Self> {
        if !self.0.storage.can_transfer_exclusively() {
            return Err(self);
        }

        let tensor = match Arc::try_unwrap(self.0) {
            Ok(tensor) => tensor,
            Err(inner) => return Err(Self(inner)),
        };
        let Tensor_ {
            id,
            storage,
            layout,
            dtype,
            device,
        } = tensor;

        match Arc::try_unwrap(storage) {
            Ok(storage) => Ok(ExclusiveTensor {
                storage,
                layout,
                dtype,
                device,
            }),
            Err(storage) => Err(Self(Arc::new(Tensor_ {
                id,
                storage,
                layout,
                dtype,
                device,
            }))),
        }
    }

    pub fn id(&self) -> TensorId {
        self.0.id
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    pub fn device(&self) -> &crate::Device {
        &self.0.device
    }

    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    pub fn shape(&self) -> &crate::Shape {
        self.layout().shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.layout().dims()
    }

    /// Returns the size of one dimension.
    pub fn dim(&self, dim: usize) -> Result<usize> {
        self.dims().get(dim).copied().ok_or(Error::InvalidDim {
            dim,
            rank: self.rank(),
        })
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

    /// Number of elements in the backing allocation, including elements
    /// outside this view's logical region.
    pub fn storage_len(&self) -> usize {
        self.storage().len()
    }

    /// Returns the backing allocation base address.
    ///
    /// The pointer is valid only while this tensor keeps the backing storage
    /// alive. For an empty storage it is dangling and must not be dereferenced.
    pub fn storage_base_ptr(&self) -> *const u8 {
        match self.storage() {
            Storage::Cpu(storage) => storage.base_ptr(),
        }
    }

    /// Returns the alignment guaranteed for the backing allocation.
    pub fn storage_alignment(&self) -> usize {
        match self.storage() {
            Storage::Cpu(storage) => storage.base_alignment(),
        }
    }

    /// Returns the guaranteed alignment of this view's first logical element.
    /// A non-zero view offset can reduce alignment even when its backing
    /// allocation remains highly aligned.
    pub fn effective_alignment(&self) -> Result<usize> {
        match self.storage() {
            Storage::Cpu(storage) => storage.effective_alignment(self.layout()),
        }
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

    /// Extracts values in logical row-major order, independent of the view's
    /// physical storage order.
    pub fn to_vec<T: WithDType>(&self) -> Result<Vec<T>> {
        let storage = self.storage();
        let cpu_storage = match &*storage {
            Storage::Cpu(storage) => storage.as_ref(),
        };
        let values = T::cpu_storage_ref_as_slice(cpu_storage)?;
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
            Storage::Cpu(storage) => storage.as_ref(),
        };
        let values = T::cpu_storage_ref_as_slice(cpu_storage)?;
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

    /// Borrows CPU storage and the tensor layout for the duration of a callback.
    pub fn with_cpu_storage<R>(
        &self,
        f: impl FnOnce(CpuStorageRef<'_>, &Layout) -> Result<R>,
    ) -> Result<R> {
        let storage = self.storage();
        let cpu_storage = match &*storage {
            Storage::Cpu(storage) => storage.as_ref(),
        };
        f(cpu_storage, self.layout())
    }

    /// Reads one element from a rank-1 tensor without constructing a scalar
    /// view or allocating a temporary vector.
    pub fn read_scalar_at<T: WithDType>(&self, index: usize) -> Result<T> {
        if self.rank() != 1 {
            return Err(Error::InvalidRank {
                expected: 1,
                actual: self.rank(),
            });
        }
        if index >= self.dims()[0] {
            return Err(Error::StorageOutOfBounds);
        }
        let physical_index = self
            .layout()
            .start_offset()
            .checked_add(
                index
                    .checked_mul(self.stride()[0])
                    .ok_or(Error::StorageOutOfBounds)?,
            )
            .ok_or(Error::StorageOutOfBounds)?;
        self.with_cpu_storage(|storage, _| {
            T::cpu_storage_ref_as_slice(storage)?
                .get(physical_index)
                .copied()
                .ok_or(Error::StorageOutOfBounds)
        })
    }

    pub fn to_vec0<T: WithDType>(&self) -> Result<T> {
        self.to_scalar()
    }

    /// Extracts the only value from a rank-0 tensor.
    pub fn to_scalar<T: WithDType>(&self) -> Result<T> {
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

    pub fn to_vec3<T: WithDType>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        if self.rank() != 3 {
            return Err(Error::InvalidRank {
                expected: 3,
                actual: self.rank(),
            });
        }
        let values = self.to_vec::<T>()?;
        let [dim0, dim1, dim2] = self.dims() else {
            unreachable!("rank was checked above");
        };
        let mut output = Vec::with_capacity(*dim0);
        let mut offset = 0usize;
        for _ in 0..*dim0 {
            let mut rows = Vec::with_capacity(*dim1);
            for _ in 0..*dim1 {
                let end = offset.checked_add(*dim2).ok_or(Error::StorageOutOfBounds)?;
                rows.push(
                    values
                        .get(offset..end)
                        .ok_or(Error::StorageOutOfBounds)?
                        .to_vec(),
                );
                offset = end;
            }
            output.push(rows);
        }
        Ok(output)
    }

    pub fn flatten_all(&self) -> Result<Self> {
        self.reshape(vec![self.elem_count()])
    }

    pub(super) fn storage(&self) -> &Storage {
        &self.0.storage
    }
}
