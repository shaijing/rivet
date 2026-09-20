use crate::error::to_py_err;
use half::{bf16, f16};
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

/// Export a CPU tensor as a read-only NumPy view.
///
/// The returned array borrows the tensor's allocation and keeps the tensor
/// alive through its base object. Unsupported devices or dtypes are reported
/// instead of silently allocating a copy; this is the current zero-copy
/// boundary for the Python API.
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
        DType::BF16 => {
            // NumPy does not ship bfloat16 on every supported version. The
            // numpy crate can use an external provider such as ml_dtypes,
            // but turn its absence into a normal Python error instead of the
            // provider lookup panicking inside Element::get_dtype.
            numpy::PyArrayDescr::new(py, "bfloat16").map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "NumPy bfloat16 support is required for a zero-copy BF16 view",
                )
            })?;
            export!(BF16, bf16)
        }
        DType::F16 => export!(F16, f16),
        DType::F32 => export!(F32, f32),
        DType::F64 => export!(F64, f64),
    }
}

fn tensor_data_pointer<T>(
    tensor: &Tensor,
    f: impl FnOnce(CpuStorageRef<'_>) -> Option<(*const T, usize)>,
) -> PyResult<*const T> {
    tensor
        .with_cpu_storage(|storage, layout| {
            let Some((pointer, storage_len)) = f(storage) else {
                return Err(rivet_core::Error::StorageOutOfBounds);
            };
            let Some((_, max_offset)) = layout.storage_bounds() else {
                // Empty views do not dereference their data pointer. Avoid
                // pointer arithmetic because an empty view may start at the
                // end of an allocation.
                return Ok(pointer);
            };
            if max_offset >= storage_len {
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
    if strides.iter().any(|&stride| stride > isize::MAX as usize)
        || tensor.elem_count() > isize::MAX as usize
    {
        return Err(to_py_err(invalid_argument(
            "tensor layout cannot be represented by NumPy",
        )));
    }
    let view = unsafe {
        // ndarray takes element strides; NumPy receives byte strides through
        // borrow_from_array below.
        ArrayViewD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), pointer)
    };
    let owner = Py::new(py, TensorArrayOwner { _tensor: tensor })?;
    let array = unsafe { PyArrayDyn::borrow_from_array(&view, owner.into_bound(py).into_any()) };
    array.readwrite().make_nonwriteable();
    Ok(array.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyUntypedArrayMethods;
    use rivet_core::Device;

    #[test]
    fn numpy_export_keeps_permute_narrow_and_broadcast_zero_copy() {
        Python::initialize();
        Python::attach(|py| {
            let permuted = Tensor::from_vec((0u8..24).collect(), (2, 3, 4), &Device::Cpu)
                .unwrap()
                .permute(&[2, 0, 1])
                .unwrap();
            let array = tensor_to_numpy(py, permuted).unwrap();
            let array = array.bind(py).cast::<PyArrayDyn<u8>>().unwrap();
            assert_eq!(array.shape(), &[4, 2, 3]);
            assert_eq!(array.strides(), &[1, 12, 4]);
            assert_eq!(
                array
                    .readonly()
                    .as_array()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![
                    0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15,
                    19, 23
                ]
            );

            let narrow = Tensor::from_vec((0i64..20).collect(), (4, 5), &Device::Cpu)
                .unwrap()
                .narrow(1, 1, 3)
                .unwrap();
            let narrow = tensor_to_numpy(py, narrow).unwrap();
            let narrow = narrow.bind(py).cast::<PyArrayDyn<i64>>().unwrap();
            assert_eq!(narrow.strides(), &[40, 8]);
            assert_eq!(
                narrow
                    .readonly()
                    .as_array()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]
            );

            let broadcast = Tensor::from_vec(vec![1f32, 2.0, 3.0], (1, 3), &Device::Cpu)
                .unwrap()
                .broadcast_as((2, 3))
                .unwrap();
            let broadcast = tensor_to_numpy(py, broadcast).unwrap();
            let broadcast = broadcast.bind(py).cast::<PyArrayDyn<f32>>().unwrap();
            assert_eq!(broadcast.strides(), &[0, 4]);
            assert_eq!(
                broadcast
                    .readonly()
                    .as_array()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
        });
    }

    #[test]
    fn numpy_export_handles_empty_layouts() {
        Python::initialize();
        Python::attach(|py| {
            let empty = Tensor::zeros((0, 3), DType::U8, &Device::Cpu).unwrap();
            let array = tensor_to_numpy(py, empty).unwrap();
            let array = array.bind(py).cast::<PyArrayDyn<u8>>().unwrap();
            assert_eq!(array.shape(), &[0, 3]);
            assert_eq!(array.strides(), &[3, 1]);
        });
    }

    #[test]
    fn numpy_export_supports_float16_without_copying() {
        Python::initialize();
        Python::attach(|py| {
            let tensor = Tensor::from_vec(
                vec![f16::from_f32(1.5), f16::from_f32(2.5)],
                [2],
                &Device::Cpu,
            )
            .unwrap();
            let array = tensor_to_numpy(py, tensor).unwrap();
            let array = array.bind(py).cast::<PyArrayDyn<f16>>().unwrap();
            assert_eq!(
                array
                    .readonly()
                    .as_array()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                [f16::from_f32(1.5), f16::from_f32(2.5)]
            );
            let flags = array.getattr("flags").unwrap();
            assert!(!flags.getattr("owndata").unwrap().extract::<bool>().unwrap());
            assert!(
                !flags
                    .getattr("writeable")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
        });
    }
}
