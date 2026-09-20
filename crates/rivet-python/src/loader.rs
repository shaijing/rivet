use crate::dataset::PyArrowDataset;
use crate::dlpack::PyDLPackTensor;
use crate::error::to_py_err;
use numpy::{
    PyArrayDyn, PyArrayMethods,
    ndarray::{ArrayViewD, IxDyn, ShapeBuilder},
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_vision::api::{
    ArrowImageDataset, ImageBatch, ImageDataLoader, ImagePipeline, empty_image_batch,
    invalid_argument,
};
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass(name = "_DataLoader")]
pub(crate) struct PyDataLoader {
    pub(crate) inner: ImageDataLoader,
}

/// Owns a tensor while a NumPy array borrows its immutable CPU allocation.
///
/// The object is installed as the ndarray's base object, so the allocation
/// cannot be released while Python still holds the view. Loader output tensors
/// are never mutated after crossing this boundary; the array is additionally
/// marked read-only to prevent Python from racing a Rust reader.
#[pyclass]
struct TensorArrayOwner {
    _tensor: Tensor,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    fn new(dataset: &PyArrowDataset, batch_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: ImagePipeline::new(Arc::clone(&dataset.inner))
                .decode_image()
                .batch(batch_size, false)
                .compile()
                .map_err(to_py_err)?,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        // Release the GIL while the rust workers load/decode/transform;
        // no python objects are touched on worker threads.
        let batch = py.detach(|| self.inner.next_batch()).map_err(to_py_err)?;

        batch.map(|batch| image_batch_to_py(py, batch)).transpose()
    }

    fn next_dlpack(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let batch = py.detach(|| self.inner.next_batch()).map_err(to_py_err)?;
        batch
            .map(|batch| image_batch_to_dlpack(py, batch))
            .transpose()
    }
}

/// Read Hugging Face Arrow IPC files and return a decoded RGB image batch.
#[pyfunction]
#[pyo3(signature = (arrow_files, batch_size, start=0, image_column="img", label_column="label"))]
pub(crate) fn read_image_batch(
    py: Python<'_>,
    arrow_files: Vec<PathBuf>,
    batch_size: usize,
    start: usize,
    image_column: &str,
    label_column: &str,
) -> PyResult<Py<PyDict>> {
    if arrow_files.is_empty() {
        return Err(to_py_err(invalid_argument("arrow_files must not be empty")));
    }

    let dataset = Arc::new(
        ArrowImageDataset::new(
            arrow_files,
            image_column.to_string(),
            label_column.to_string(),
        )
        .map_err(to_py_err)?,
    );
    let mut loader = ImagePipeline::new(dataset)
        .decode_image()
        .batch(batch_size, false)
        .compile_from(start)
        .map_err(to_py_err)?;
    let batch = loader
        .next_batch()
        .map_err(to_py_err)?
        .unwrap_or_else(|| empty_image_batch().expect("empty batch"));

    image_batch_to_py(py, batch)
}

pub(super) fn image_batch_to_py(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let ImageBatch { images, labels } = batch;
    let dtype = images.dtype();
    let shape = images.dims().to_vec();
    let images = image_array_to_py(py, images)?;
    let labels = tensor_array_to_py(py, labels)?;

    let out = PyDict::new(py);
    out.set_item("images", images)?;
    out.set_item("labels", labels)?;
    out.set_item("shape", &shape)?;
    out.set_item("dtype", dtype_name(dtype))?;
    out.set_item("layout", batch_layout(&shape))?;

    Ok(out.into())
}

fn image_batch_to_dlpack(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let ImageBatch { images, labels } = batch;
    let out = PyDict::new(py);
    out.set_item("images", Py::new(py, PyDLPackTensor::new(images))?)?;
    out.set_item("labels", Py::new(py, PyDLPackTensor::new(labels))?)?;
    Ok(out.into())
}

fn image_array_to_py(py: Python<'_>, images: Tensor) -> PyResult<Py<PyAny>> {
    tensor_array_to_py(py, images)
}

fn tensor_array_to_py(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    match tensor.dtype() {
        DType::U8 => tensor_array_view_u8(py, tensor),
        DType::U32 => tensor_array_view_u32(py, tensor),
        DType::I16 => tensor_array_view_i16(py, tensor),
        DType::I32 => tensor_array_view_i32(py, tensor),
        DType::F32 => tensor_array_view_f32(py, tensor),
        DType::F64 => tensor_array_view_f64(py, tensor),
        DType::I64 => tensor_array_view_i64(py, tensor),
        dtype => Err(to_py_err(invalid_argument(format!(
            "Python tensor conversion does not support {:?}",
            dtype
        )))),
    }
}

fn tensor_array_view_u8(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::U8(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<u8>(py, tensor, pointer)
}

fn tensor_array_view_u32(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::U32(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<u32>(py, tensor, pointer)
}

fn tensor_array_view_i16(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::I16(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<i16>(py, tensor, pointer)
}

fn tensor_array_view_i32(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::I32(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<i32>(py, tensor, pointer)
}

fn tensor_array_view_f32(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::F32(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<f32>(py, tensor, pointer)
}

fn tensor_array_view_i64(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::I64(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<i64>(py, tensor, pointer)
}

fn tensor_array_view_f64(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = tensor_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::F64(values) => Some((values.as_ptr(), values.len())),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<f64>(py, tensor, pointer)
}

fn tensor_data_pointer<T>(
    tensor: &Tensor,
    storage_pointer: impl FnOnce(CpuStorageRef<'_>) -> Option<(*const T, usize)>,
) -> PyResult<*const T> {
    tensor
        .with_cpu_storage(|storage, layout| {
            let Some((pointer, storage_len)) = storage_pointer(storage) else {
                return Err(rivet_core::Error::StorageOutOfBounds);
            };
            if layout.elem_count() == 0 {
                return Ok(pointer);
            }
            let max_offset = layout.dims().iter().zip(layout.stride()).try_fold(
                layout.start_offset(),
                |max_offset, (&dim, &stride)| {
                    let span = dim
                        .checked_sub(1)
                        .and_then(|extent| extent.checked_mul(stride))
                        .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                    max_offset
                        .checked_add(span)
                        .ok_or(rivet_core::Error::StorageOutOfBounds)
                },
            )?;
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
        // Layout bounds were verified against the immutable backing allocation
        // above. ndarray takes element strides; NumPy receives byte strides
        // through borrow_from_array below. TensorArrayOwner keeps allocation
        // alive for the exported array's complete lifetime.
        ArrayViewD::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&strides)), pointer)
    };
    let owner = Py::new(py, TensorArrayOwner { _tensor: tensor })?;
    let array = unsafe { PyArrayDyn::borrow_from_array(&view, owner.into_bound(py).into_any()) };
    array.readwrite().make_nonwriteable();
    Ok(array.into_any().unbind())
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::F32 => "float32",
        _ => "unsupported",
    }
}

fn batch_layout(shape: &[usize]) -> &'static str {
    if shape.len() == 4 && shape[1] == 3 {
        "NCHW"
    } else {
        "NHWC"
    }
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
            let array = tensor_array_to_py(py, permuted).unwrap();
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
            let narrow = tensor_array_to_py(py, narrow).unwrap();
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
            let broadcast = tensor_array_to_py(py, broadcast).unwrap();
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
            let array = tensor_array_to_py(py, empty).unwrap();
            let array = array.bind(py).cast::<PyArrayDyn<u8>>().unwrap();
            assert_eq!(array.shape(), &[0, 3]);
            assert_eq!(array.strides(), &[3, 1]);
        });
    }
}
