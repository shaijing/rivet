use crate::dtype;
use pyo3::{ffi, prelude::*};
use rivet_core::{ExclusiveTensor, Tensor};
use std::ffi::{c_char, c_void};

const DLTENSOR: &[u8] = b"dltensor\0";
const DLTENSOR_VERSIONED: &[u8] = b"dltensor_versioned\0";
const DL_DEVICE_CPU: i32 = 1;
const DLPACK_MAJOR_VERSION: u32 = 1;
const DLPACK_MINOR_VERSION: u32 = 0;
const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;

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
    _owner: TensorOwner,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}
#[repr(C)]
struct ManagedTensorVersioned {
    managed: DLManagedTensorVersioned,
    _owner: TensorOwner,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

enum TensorOwner {
    /// Standard DLPack is a borrowed, read-only shared export.
    Shared { _tensor: Tensor },
    /// Explicit `into_dlpack` is a mutable ownership transfer.
    Exclusive { _tensor: ExclusiveTensor },
}

#[derive(Clone, Copy)]
enum DlpackAbi {
    Legacy,
    Versioned,
}

struct DlpackStorageView {
    base_ptr: *const u8,
    byte_offset: u64,
}

struct DlpackLayout {
    ndim: i32,
    shape: Vec<i64>,
    strides: Vec<i64>,
    data: *mut c_void,
    byte_offset: u64,
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

fn cpu_storage_view(tensor: &Tensor) -> rivet_core::Result<DlpackStorageView> {
    let itemsize = tensor.dtype().size_in_bytes();
    let base_ptr = tensor.storage_base_ptr();
    let storage_len = tensor.storage_len();
    tensor.with_cpu_storage(|_, layout| {
        if layout.elem_count() == 0 {
            // Empty views do not dereference their data pointer. In
            // particular, their start offset may legally be one-past-end.
            return Ok(DlpackStorageView {
                base_ptr,
                byte_offset: 0,
            });
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
        Ok(DlpackStorageView {
            base_ptr,
            byte_offset: u64::try_from(byte_offset)
                .map_err(|_| rivet_core::Error::StorageOutOfBounds)?,
        })
    })
}

fn dlpack_layout(tensor: &Tensor) -> PyResult<DlpackLayout> {
    let shape: Vec<i64> = tensor
        .dims()
        .iter()
        .map(|value| i64::try_from(*value))
        .collect::<Result<_, _>>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("shape exceeds DLPack range"))?;
    let strides: Vec<i64> = tensor
        .stride()
        .iter()
        .map(|value| i64::try_from(*value))
        .collect::<Result<_, _>>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("stride exceeds DLPack range"))?;
    let ndim = i32::try_from(shape.len())
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rank exceeds DLPack range"))?;
    let storage = cpu_storage_view(tensor)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let (data, byte_offset) = if tensor.elem_count() == 0 {
        (std::ptr::null_mut(), 0)
    } else {
        (storage.base_ptr.cast_mut().cast(), storage.byte_offset)
    };
    Ok(DlpackLayout {
        ndim,
        shape,
        strides,
        data,
        byte_offset,
    })
}

fn negotiate_dlpack_abi(max_version: Option<(u32, u32)>) -> PyResult<DlpackAbi> {
    let Some((major, minor)) = max_version else {
        return Ok(DlpackAbi::Legacy);
    };
    if (major, minor) < (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Rivet DLPack export supports version 1.0; max_version is too old",
        ));
    }
    Ok(DlpackAbi::Versioned)
}

fn validate_dlpack_request(
    stream: Option<i64>,
    max_version: Option<(u32, u32)>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<DlpackAbi> {
    if let Some(stream) = stream
        && stream != 0
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Rivet CPU DLPack export only supports stream=None or stream=0",
        ));
    }
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
    negotiate_dlpack_abi(max_version)
}

fn py_capsule_result(py: Python<'_>, capsule: *mut ffi::PyObject) -> PyResult<Py<PyAny>> {
    if capsule.is_null() {
        return Err(PyErr::fetch(py));
    }
    unsafe { Bound::from_owned_ptr_or_err(py, capsule) }.map(|value| value.into_any().unbind())
}

#[pyclass(name = "_DLPackTensor")]
pub(crate) struct PyDLPackTensor {
    /// Standard DLPack exports clone this cheap Tensor handle. Only the
    /// explicit into_dlpack path takes it and consumes the producer.
    tensor: Option<Tensor>,
}

impl PyDLPackTensor {
    pub(crate) fn new(tensor: Tensor) -> Self {
        Self {
            tensor: Some(tensor),
        }
    }

    fn consumed_error() -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(
            "DLPack producer has already transferred ownership",
        )
    }

    fn export_owned(
        &mut self,
        py: Python<'_>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
        read_only: bool,
    ) -> PyResult<Py<PyAny>> {
        let tensor = self.tensor.as_ref().ok_or_else(Self::consumed_error)?;
        let abi = validate_dlpack_request(stream, max_version, dl_device, copy)?;
        let DlpackLayout {
            ndim,
            shape,
            strides,
            data,
            byte_offset,
        } = dlpack_layout(tensor)?;
        let (code, bits) = dtype::dlpack_code_bits(tensor.dtype());
        let tensor = self.tensor.take().ok_or_else(Self::consumed_error)?;
        let owner = if read_only {
            TensorOwner::Shared { _tensor: tensor }
        } else {
            match tensor.try_into_exclusive() {
                Ok(exclusive) => TensorOwner::Exclusive { _tensor: exclusive },
                Err(tensor) => {
                    let message = if !tensor.is_uniquely_owned() {
                        "cannot transfer aliased Rivet storage without copying"
                    } else {
                        "cannot transfer read-only or externally owned Rivet storage without copying"
                    };
                    self.tensor = Some(tensor);
                    return Err(pyo3::exceptions::PyBufferError::new_err(message));
                }
            }
        };

        if matches!(abi, DlpackAbi::Versioned) {
            let mut owner = Box::new(ManagedTensorVersioned {
                managed: DLManagedTensorVersioned {
                    version: DLPackVersion {
                        major: DLPACK_MAJOR_VERSION,
                        minor: DLPACK_MINOR_VERSION,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: Some(dlpack_versioned_deleter),
                    flags: if read_only {
                        DLPACK_FLAG_BITMASK_READ_ONLY
                    } else {
                        0
                    },
                    dl_tensor: DLTensor {
                        data,
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
                        byte_offset,
                    },
                },
                _owner: owner,
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
                        data,
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
                        byte_offset,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: Some(dlpack_deleter),
                },
                _owner: owner,
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

    fn export_shared(
        &self,
        py: Python<'_>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let tensor = self
            .tensor
            .as_ref()
            .ok_or_else(Self::consumed_error)?
            .clone();
        let mut export = Self::new(tensor);
        export.export_owned(py, stream, max_version, dl_device, copy, true)
    }
}

#[pymethods]
impl PyDLPackTensor {
    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        if self.tensor.is_none() {
            return Err(Self::consumed_error());
        }
        Ok((DL_DEVICE_CPU, 0))
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
        self.export_shared(py, stream, max_version, dl_device, copy)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn into_dlpack(
        &mut self,
        py: Python<'_>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        self.export_owned(py, stream, max_version, dl_device, copy, false)
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
            let base_pointer = base.storage_base_ptr().cast_mut().cast();
            assert_eq!(base.storage_alignment(), rivet_core::CPU_STORAGE_ALIGNMENT);
            assert_eq!(permuted.data, base_pointer);
            assert_eq!(permuted.byte_offset, 0);
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
            assert_eq!(narrow.data, base_pointer);
            assert_eq!(narrow.byte_offset, 4);
            drop(narrow_capsule);

            let f32_base =
                Tensor::from_vec((0..8).map(|value| value as f32).collect(), 8, &Device::Cpu)
                    .unwrap();
            let f32_view = f32_base.narrow(0, 1, 4).unwrap();
            assert_eq!(f32_view.storage_base_ptr(), f32_base.storage_base_ptr());
            assert_eq!(f32_view.effective_alignment().unwrap(), 4);
            let (f32_capsule, f32_tensor) = capsule_tensor(py, f32_view);
            assert_eq!(
                f32_tensor.data,
                f32_base.storage_base_ptr().cast_mut().cast()
            );
            assert_eq!(f32_tensor.byte_offset, std::mem::size_of::<f32>() as u64);
            drop(f32_capsule);

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
            assert!(empty.data.is_null());
            assert_eq!(empty.byte_offset, 0);
            drop(empty_capsule);
        });
    }

    #[test]
    fn dlpack_validates_stream_device_and_version() {
        Python::initialize();
        Python::attach(|py| {
            let producer =
                PyDLPackTensor::new(Tensor::zeros((2,), DType::U8, &Device::Cpu).unwrap());
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

            let accepted =
                PyDLPackTensor::new(Tensor::zeros((2,), DType::U8, &Device::Cpu).unwrap());
            let accepted_capsule = accepted
                .__dlpack__(
                    py,
                    Some(0),
                    Some((1, 0)),
                    Some((DL_DEVICE_CPU, 0)),
                    Some(false),
                )
                .unwrap();
            drop(accepted_capsule);

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
            assert_eq!(unsafe { (*managed).flags }, DLPACK_FLAG_BITMASK_READ_ONLY);

            let repeated = producer
                .__dlpack__(py, None, Some((1, 0)), None, None)
                .unwrap();
            drop(repeated);
        });
    }

    #[test]
    fn into_dlpack_rejects_storage_aliases_until_they_are_dropped() {
        Python::initialize();
        Python::attach(|py| {
            let base = Tensor::from_vec(vec![0u8, 1, 2, 3], (2, 2), &Device::Cpu).unwrap();
            let view = base.narrow(0, 1, 1).unwrap();
            let mut producer = PyDLPackTensor::new(base);

            let error = producer
                .into_dlpack(py, None, None, None, None)
                .expect_err("aliased storage must not become a writable DLPack export");
            assert!(error.to_string().contains("aliased"));

            drop(view);
            let capsule = producer.into_dlpack(py, None, None, None, None).unwrap();
            drop(capsule);
        });
    }
}
