use super::{Tensor, TensorId};
use crate::cpu_backend::CpuStorageRef;
use crate::storage::Storage;
use crate::{DType, Error, Layout, Result, WithDType};
use std::sync::Arc;

impl Tensor {
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
