use pyo3::{ffi, prelude::*};
use rivet_core::{CpuStorageRef, DType, Tensor};
use std::ffi::{c_char, c_void};

const DLTENSOR: &[u8] = b"dltensor\0";
const DL_DEVICE_CPU: i32 = 1;
const DL_INT: u8 = 0;
const DL_UINT: u8 = 1;
const DL_FLOAT: u8 = 2;

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}
#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
struct ManagedTensor {
    managed: DLManagedTensor,
    _tensor: Tensor,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    if !managed.is_null() {
        unsafe { drop(Box::from_raw(managed.cast::<ManagedTensor>())) };
    }
}
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    if unsafe { ffi::PyCapsule_IsValid(capsule, DLTENSOR.as_ptr().cast::<c_char>()) } != 0 {
        let managed =
            unsafe { ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr().cast::<c_char>()) }
                .cast::<DLManagedTensor>();
        unsafe { dlpack_deleter(managed) };
    }
}

#[pyclass(name = "_DLPackTensor")]
pub(crate) struct PyDLPackTensor {
    tensor: Tensor,
}

impl PyDLPackTensor {
    pub(crate) fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }
}

#[pymethods]
impl PyDLPackTensor {
    fn __dlpack_device__(&self) -> (i32, i32) {
        (DL_DEVICE_CPU, 0)
    }
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (stream, max_version, dl_device);
        if copy == Some(true) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Rivet DLPack export is zero-copy only",
            ));
        }
        let shape: Vec<i64> = self
            .tensor
            .dims()
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<Result<_, _>>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("shape exceeds DLPack range"))?;
        let strides: Vec<i64> = self
            .tensor
            .stride()
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<Result<_, _>>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("stride exceeds DLPack range"))?;
        let (code, bits) = match self.tensor.dtype() {
            DType::U8 => (DL_UINT, 8),
            DType::U32 => (DL_UINT, 32),
            DType::I16 => (DL_INT, 16),
            DType::I32 => (DL_INT, 32),
            DType::I64 => (DL_INT, 64),
            DType::F32 => (DL_FLOAT, 32),
            DType::F64 => (DL_FLOAT, 64),
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "dtype is not supported by Rivet DLPack export",
                ));
            }
        };
        let data = self
            .tensor
            .with_cpu_storage(|storage, layout| {
                let p: *const u8 = match storage {
                    CpuStorageRef::U8(v) => v.as_ptr().cast(),
                    CpuStorageRef::U32(v) => v.as_ptr().cast(),
                    CpuStorageRef::I16(v) => v.as_ptr().cast(),
                    CpuStorageRef::I32(v) => v.as_ptr().cast(),
                    CpuStorageRef::I64(v) => v.as_ptr().cast(),
                    CpuStorageRef::F32(v) => v.as_ptr().cast(),
                    CpuStorageRef::F64(v) => v.as_ptr().cast(),
                    _ => return Err(rivet_core::Error::StorageOutOfBounds),
                };
                Ok(if layout.elem_count() == 0 {
                    p
                } else {
                    unsafe { p.add(layout.start_offset() * self.tensor.dtype().size_in_bytes()) }
                })
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut owner = Box::new(ManagedTensor {
            managed: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data.cast_mut().cast(),
                    device: DLDevice {
                        device_type: DL_DEVICE_CPU,
                        device_id: 0,
                    },
                    ndim: i32::try_from(shape.len()).unwrap(),
                    dtype: DLDataType {
                        code,
                        bits,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(dlpack_deleter),
            },
            _tensor: self.tensor.clone(),
            _shape: shape,
            _strides: strides,
        });
        owner.managed.dl_tensor.shape = owner._shape.as_mut_ptr();
        owner.managed.dl_tensor.strides = owner._strides.as_mut_ptr();
        let raw = Box::into_raw(owner)
            .cast::<DLManagedTensor>()
            .cast::<c_void>();
        let capsule = unsafe {
            ffi::PyCapsule_New(
                raw,
                DLTENSOR.as_ptr().cast::<c_char>(),
                Some(capsule_destructor),
            )
        };
        unsafe { Bound::from_owned_ptr_or_err(py, capsule) }.map(|v| v.into_any().unbind())
    }
}
