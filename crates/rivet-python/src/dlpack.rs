use crate::dtype;
use pyo3::{ffi, prelude::*};
use rivet_core::{CpuStorageRef, Tensor};
use std::ffi::{c_char, c_void};

const DLTENSOR: &[u8] = b"dltensor\0";
const DLTENSOR_VERSIONED: &[u8] = b"dltensor_versioned\0";
const DL_DEVICE_CPU: i32 = 1;
const DLPACK_MAJOR_VERSION: u32 = 1;
const DLPACK_MINOR_VERSION: u32 = 0;

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
struct DLPackVersion {
    major: u32,
    minor: u32,
}
#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

#[repr(C)]
struct ManagedTensor {
    managed: DLManagedTensor,
    _tensor: Tensor,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}
#[repr(C)]
struct ManagedTensorVersioned {
    managed: DLManagedTensorVersioned,
    _tensor: Tensor,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    if !managed.is_null() {
        unsafe { drop(Box::from_raw(managed.cast::<ManagedTensor>())) };
    }
}

unsafe extern "C" fn dlpack_versioned_deleter(managed: *mut DLManagedTensorVersioned) {
    if !managed.is_null() {
        unsafe { drop(Box::from_raw(managed.cast::<ManagedTensorVersioned>())) };
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    if unsafe { ffi::PyCapsule_IsValid(capsule, DLTENSOR.as_ptr().cast::<c_char>()) } != 0 {
        let managed = unsafe {
            ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr().cast::<c_char>())
                .cast::<DLManagedTensor>()
        };
        unsafe { dlpack_deleter(managed) };
    } else if unsafe {
        ffi::PyCapsule_IsValid(capsule, DLTENSOR_VERSIONED.as_ptr().cast::<c_char>())
    } != 0
    {
        let managed = unsafe {
            ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED.as_ptr().cast::<c_char>())
                .cast::<DLManagedTensorVersioned>()
        };
        unsafe { dlpack_versioned_deleter(managed) };
    }
}

fn cpu_data_pointer(tensor: &Tensor) -> rivet_core::Result<*const u8> {
    let itemsize = tensor.dtype().size_in_bytes();
    tensor.with_cpu_storage(|storage, layout| {
        let (pointer, storage_len): (*const u8, usize) = match storage {
            CpuStorageRef::U8(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::U32(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::I16(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::I32(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::I64(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::BF16(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::F16(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::F32(values) => (values.as_ptr().cast(), values.len()),
            CpuStorageRef::F64(values) => (values.as_ptr().cast(), values.len()),
        };

        if layout.elem_count() == 0 {
            // Empty views do not dereference their data pointer. In
            // particular, their start offset may legally be one-past-end.
            return Ok(pointer);
        }

        let Some((start_offset, max_offset)) = layout.storage_bounds() else {
            return Err(rivet_core::Error::StorageOutOfBounds);
        };
        if max_offset >= storage_len {
            return Err(rivet_core::Error::StorageOutOfBounds);
        }
        let storage_bytes = storage_len
            .checked_mul(itemsize)
            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
        let byte_offset = start_offset
            .checked_mul(itemsize)
            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
        let end_offset = max_offset
            .checked_add(1)
            .and_then(|offset| offset.checked_mul(itemsize))
            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
        if end_offset > storage_bytes || byte_offset >= storage_bytes {
            return Err(rivet_core::Error::StorageOutOfBounds);
        }
        Ok(unsafe { pointer.add(byte_offset) })
    })
}

fn py_capsule_result(py: Python<'_>, capsule: *mut ffi::PyObject) -> PyResult<Py<PyAny>> {
    if capsule.is_null() {
        return Err(PyErr::fetch(py));
    }
    unsafe { Bound::from_owned_ptr_or_err(py, capsule) }.map(|value| value.into_any().unbind())
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
        if let Some(stream) = stream
            && stream != 0
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Rivet CPU DLPack export only supports stream=None or stream=0",
            ));
        }
        let versioned = if let Some((major, minor)) = max_version {
            if (major, minor) < (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Rivet DLPack export supports version 1.0; max_version is too old",
                ));
            }
            true
        } else {
            false
        };
        if copy == Some(true) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Rivet DLPack export is zero-copy only; copy=True is unsupported",
            ));
        }
        if let Some((device_type, device_id)) = dl_device
            && (device_type != DL_DEVICE_CPU || device_id != 0)
        {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Rivet currently exports DLPack only on CPU device 0",
            ));
        }

        let shape: Vec<i64> = self
            .tensor
            .dims()
            .iter()
            .map(|value| i64::try_from(*value))
            .collect::<Result<_, _>>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("shape exceeds DLPack range"))?;
        let strides: Vec<i64> = self
            .tensor
            .stride()
            .iter()
            .map(|value| i64::try_from(*value))
            .collect::<Result<_, _>>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("stride exceeds DLPack range"))?;
        let ndim = i32::try_from(shape.len())
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("rank exceeds DLPack range"))?;
        let (code, bits) = dtype::dlpack_code_bits(self.tensor.dtype());
        let data = cpu_data_pointer(&self.tensor)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;

        if versioned {
            let mut owner = Box::new(ManagedTensorVersioned {
                managed: DLManagedTensorVersioned {
                    version: DLPackVersion {
                        major: DLPACK_MAJOR_VERSION,
                        minor: DLPACK_MINOR_VERSION,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: Some(dlpack_versioned_deleter),
                    flags: 0,
                    dl_tensor: DLTensor {
                        data: data.cast_mut().cast(),
                        device: DLDevice {
                            device_type: DL_DEVICE_CPU,
                            device_id: 0,
                        },
                        ndim,
                        dtype: DLDataType {
                            code,
                            bits,
                            lanes: 1,
                        },
                        shape: std::ptr::null_mut(),
                        strides: std::ptr::null_mut(),
                        byte_offset: 0,
                    },
                },
                _tensor: self.tensor.clone(),
                _shape: shape,
                _strides: strides,
            });
            owner.managed.dl_tensor.shape = owner._shape.as_mut_ptr();
            owner.managed.dl_tensor.strides = owner._strides.as_mut_ptr();
            let raw = Box::into_raw(owner);
            let capsule = unsafe {
                ffi::PyCapsule_New(
                    raw.cast::<c_void>(),
                    DLTENSOR_VERSIONED.as_ptr().cast::<c_char>(),
                    Some(capsule_destructor),
                )
            };
            if capsule.is_null() {
                unsafe { dlpack_versioned_deleter(raw.cast()) };
                return Err(PyErr::fetch(py));
            }
            py_capsule_result(py, capsule)
        } else {
            let mut owner = Box::new(ManagedTensor {
                managed: DLManagedTensor {
                    dl_tensor: DLTensor {
                        data: data.cast_mut().cast(),
                        device: DLDevice {
                            device_type: DL_DEVICE_CPU,
                            device_id: 0,
                        },
                        ndim,
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
            let raw = Box::into_raw(owner);
            let capsule = unsafe {
                ffi::PyCapsule_New(
                    raw.cast::<c_void>(),
                    DLTENSOR.as_ptr().cast::<c_char>(),
                    Some(capsule_destructor),
                )
            };
            if capsule.is_null() {
                unsafe { dlpack_deleter(raw.cast()) };
                return Err(PyErr::fetch(py));
            }
            py_capsule_result(py, capsule)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::{DType, Device};

    fn capsule_tensor<'py>(py: Python<'py>, tensor: Tensor) -> (Py<PyAny>, &'py DLTensor) {
        let producer = PyDLPackTensor::new(tensor);
        let capsule = producer.__dlpack__(py, None, None, None, None).unwrap();
        let managed = unsafe {
            ffi::PyCapsule_GetPointer(
                capsule.bind(py).as_ptr(),
                DLTENSOR.as_ptr().cast::<c_char>(),
            )
            .cast::<DLManagedTensor>()
        };
        assert!(!managed.is_null());
        let tensor = unsafe { &(*managed).dl_tensor };
        (capsule, tensor)
    }

    #[test]
    fn dlpack_preserves_view_layouts_and_empty_shapes() {
        Python::initialize();
        Python::attach(|py| {
            let base = Tensor::from_vec((0u8..24).collect(), (2, 3, 4), &Device::Cpu).unwrap();

            let (permuted_capsule, permuted) =
                capsule_tensor(py, base.clone().permute(&[2, 0, 1]).unwrap());
            assert_eq!(permuted.ndim, 3);
            assert_eq!(
                unsafe { std::slice::from_raw_parts(permuted.shape, 3) },
                &[4, 2, 3]
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(permuted.strides, 3) },
                &[1, 12, 4]
            );
            let base_pointer = base
                .with_cpu_storage(|storage, _| match storage {
                    CpuStorageRef::U8(values) => Ok(values.as_ptr().cast_mut().cast()),
                    _ => unreachable!(),
                })
                .unwrap();
            assert_eq!(permuted.data, base_pointer);
            drop(permuted_capsule);

            let (narrow_capsule, narrow) = capsule_tensor(py, base.narrow(1, 1, 2).unwrap());
            assert_eq!(
                unsafe { std::slice::from_raw_parts(narrow.shape, 3) },
                &[2, 2, 4]
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(narrow.strides, 3) },
                &[12, 4, 1]
            );
            assert_eq!(narrow.data, unsafe { base_pointer.byte_add(4) });
            drop(narrow_capsule);

            let broadcast = Tensor::from_vec(vec![1u8, 2, 3], (1, 3), &Device::Cpu)
                .unwrap()
                .broadcast_as((2, 3))
                .unwrap();
            let (broadcast_capsule, broadcast) = capsule_tensor(py, broadcast);
            assert_eq!(
                unsafe { std::slice::from_raw_parts(broadcast.shape, 2) },
                &[2, 3]
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(broadcast.strides, 2) },
                &[0, 1]
            );
            drop(broadcast_capsule);

            let (empty_capsule, empty) =
                capsule_tensor(py, Tensor::zeros((0, 3), DType::U8, &Device::Cpu).unwrap());
            assert_eq!(
                unsafe { std::slice::from_raw_parts(empty.shape, 2) },
                &[0, 3]
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(empty.strides, 2) },
                &[3, 1]
            );
            drop(empty_capsule);
        });
    }

    #[test]
    fn dlpack_validates_stream_device_and_version() {
        Python::initialize();
        Python::attach(|py| {
            let producer =
                PyDLPackTensor::new(Tensor::zeros((2,), DType::U8, &Device::Cpu).unwrap());
            assert!(
                producer
                    .__dlpack__(py, Some(0), None, Some((DL_DEVICE_CPU, 0)), Some(false))
                    .is_ok()
            );
            assert!(producer.__dlpack__(py, Some(1), None, None, None).is_err());
            assert!(
                producer
                    .__dlpack__(py, None, None, Some((2, 0)), None)
                    .is_err()
            );
            assert!(
                producer
                    .__dlpack__(py, None, Some((0, 9)), None, None)
                    .is_err()
            );

            let capsule = producer
                .__dlpack__(py, None, Some((1, 0)), None, None)
                .unwrap();
            let managed = unsafe {
                ffi::PyCapsule_GetPointer(
                    capsule.bind(py).as_ptr(),
                    DLTENSOR_VERSIONED.as_ptr().cast::<c_char>(),
                )
                .cast::<DLManagedTensorVersioned>()
            };
            assert_eq!(unsafe { (*managed).version.major }, DLPACK_MAJOR_VERSION);
            assert_eq!(unsafe { (*managed).version.minor }, DLPACK_MINOR_VERSION);
        });
    }
}
