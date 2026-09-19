use crate::{DType, Layout, Result, Shape, WithDType};

/// Minimal storage boundary used by the tensor object and future backends.
pub trait BackendStorage: Send + Sync + Sized + 'static {
    type Device: BackendDevice<Storage = Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> &Self::Device;

    fn try_clone(&self, layout: &Layout) -> Result<Self>;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self>;
}

/// Minimal device boundary for allocating backend storage.
pub trait BackendDevice: Clone + Send + Sync + 'static {
    type Storage: BackendStorage<Device = Self>;

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    fn ones(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    fn storage_from_vec<T: WithDType>(&self, data: Vec<T>) -> Result<Self::Storage>;

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage>;
}
