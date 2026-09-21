pub(crate) mod buffer;
pub(crate) mod device;
pub(crate) mod dispatch;
pub(crate) mod index;
pub(crate) mod math;
pub(crate) mod matmul;
pub(crate) mod storage;
pub mod utils;

pub use buffer::CPU_STORAGE_ALIGNMENT;
pub use device::CpuDevice;
pub use storage::{CpuStorage, CpuStorageMutRef, CpuStorageRef};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;
    use crate::cpu_backend::buffer::CPU_STORAGE_ALIGNMENT;
    use crate::{DType, Shape};

    fn assert_aligned(storage: &CpuStorage) {
        match storage {
            CpuStorage::U8(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::U32(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::I16(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::I32(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::I64(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::BF16(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::F16(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::F32(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
            CpuStorage::F64(data) => assert!(data.is_aligned_to(CPU_STORAGE_ALIGNMENT)),
        }
    }

    #[test]
    fn all_owned_cpu_storage_variants_are_aligned() {
        let device = CpuDevice;
        for dtype in [
            DType::U8,
            DType::U32,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
        ] {
            assert_aligned(&device.zeros(&Shape::from(17), dtype).unwrap());
        }
    }
}
