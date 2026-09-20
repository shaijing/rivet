use crate::error::to_py_err;
use numpy::{
    PyArrayDyn, PyArrayMethods,
    ndarray::{ArrayViewD, IxDyn, ShapeBuilder},
};
use pyo3::prelude::*;
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_vision::api::invalid_argument;

#[pyclass]
struct TensorArrayOwner {
    _tensor: Tensor,
}

pub(crate) fn tensor_to_numpy(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    macro_rules! export {
        ($variant:ident, $ty:ty) => {{
            let pointer = tensor_data_pointer(&tensor, |storage| match storage {
                CpuStorageRef::$variant(values) => Some((values.as_ptr(), values.len())),
                _ => None,
            })?;
            tensor_array_view_from_pointer::<$ty>(py, tensor, pointer)
        }};
    }
    match tensor.dtype() {
        DType::U8 => export!(U8, u8),
        DType::U32 => export!(U32, u32),
        DType::I16 => export!(I16, i16),
        DType::I32 => export!(I32, i32),
        DType::I64 => export!(I64, i64),
        DType::F32 => export!(F32, f32),
        DType::F64 => export!(F64, f64),
        dtype => Err(to_py_err(invalid_argument(format!(
            "Python tensor conversion does not support {dtype:?}"
        )))),
    }
}

fn tensor_data_pointer<T>(
    tensor: &Tensor,
    f: impl FnOnce(CpuStorageRef<'_>) -> Option<(*const T, usize)>,
) -> PyResult<*const T> {
    tensor
        .with_cpu_storage(|storage, layout| {
            let Some((pointer, len)) = f(storage) else {
                return Err(rivet_core::Error::StorageOutOfBounds);
            };
            if layout.elem_count() == 0 {
                return Ok(pointer);
            }
            let max = layout.dims().iter().zip(layout.stride()).try_fold(
                layout.start_offset(),
                |offset, (&dim, &stride)| {
                    offset
                        .checked_add(
                            dim.checked_sub(1)
                                .and_then(|n| n.checked_mul(stride))
                                .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                        )
                        .ok_or(rivet_core::Error::StorageOutOfBounds)
                },
            )?;
            if max >= len {
                return Err(rivet_core::Error::StorageOutOfBounds);
            }
            Ok(unsafe { pointer.add(layout.start_offset()) })
        })
        .map_err(|err| to_py_err(invalid_argument(err)))
}

fn tensor_array_view_from_pointer<T: numpy::Element>(
    py: Python<'_>,
    tensor: Tensor,
    pointer: *const T,
) -> PyResult<Py<PyAny>> {
    let shape = tensor.dims().to_vec();
    let strides = tensor.stride().to_vec();
    if strides.iter().any(|&s| s > isize::MAX as usize) || tensor.elem_count() > isize::MAX as usize
    {
        return Err(to_py_err(invalid_argument(
            "tensor layout cannot be represented by NumPy",
        )));
    }
    let view =
        unsafe { ArrayViewD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), pointer) };
    let owner = Py::new(py, TensorArrayOwner { _tensor: tensor })?;
    let array = unsafe { PyArrayDyn::borrow_from_array(&view, owner.into_bound(py).into_any()) };
    array.readwrite().make_nonwriteable();
    Ok(array.into_any().unbind())
}
