pub mod backend;
#[cfg(feature = "bench-internals")]
pub mod bench;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod ops;
pub mod shape;
pub mod storage;
pub mod strided_index;
pub mod tensor;

pub use cpu_backend::CPU_STORAGE_ALIGNMENT;
pub use cpu_backend::{CpuDevice, CpuStorage, CpuStorageMutRef, CpuStorageRef};
#[cfg(feature = "cuda")]
pub use cuda_backend::CudaDevice;
pub use device::{Device, DeviceLocation};
pub use dtype::{DType, ExactOutput, WithDType};
pub use error::{Error, Result};
pub use layout::Layout;
pub use ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
pub use shape::Shape;
pub use storage::Storage;
pub use strided_index::StridedIndex;
pub use tensor::{ExclusiveTensor, RangeElement, Tensor, TensorId};
