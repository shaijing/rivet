use crate::dataset::PyArrowDataset;
use crate::error::to_py_err;
use numpy::{
    PyArray1, PyArrayDyn, PyArrayMethods,
    ndarray::{ArrayViewD, IxDyn},
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

fn image_array_to_py(py: Python<'_>, images: Tensor) -> PyResult<Py<PyAny>> {
    tensor_array_to_py(py, images)
}

fn tensor_array_to_py(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    if !tensor.is_contiguous() {
        return tensor_array_copy_to_py(py, tensor);
    }

    match tensor.dtype() {
        DType::U8 => tensor_array_view_u8(py, tensor),
        DType::F32 => tensor_array_view_f32(py, tensor),
        DType::I64 => tensor_array_view_i64(py, tensor),
        dtype => Err(to_py_err(invalid_argument(format!(
            "Python tensor conversion does not support {:?}",
            dtype
        )))),
    }
}

fn tensor_array_view_u8(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = contiguous_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::U8(values) => Some(values.as_ptr()),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<u8>(py, tensor, pointer)
}

fn tensor_array_view_f32(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = contiguous_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::F32(values) => Some(values.as_ptr()),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<f32>(py, tensor, pointer)
}

fn tensor_array_view_i64(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let pointer = contiguous_data_pointer(&tensor, |storage| match storage {
        CpuStorageRef::I64(values) => Some(values.as_ptr()),
        _ => None,
    })?;
    tensor_array_view_from_pointer::<i64>(py, tensor, pointer)
}

fn contiguous_data_pointer<T>(
    tensor: &Tensor,
    storage_pointer: impl FnOnce(CpuStorageRef<'_>) -> Option<*const T>,
) -> PyResult<*const T> {
    tensor
        .with_cpu_storage(|storage, layout| {
            let Some((start, _)) = layout.contiguous_offsets() else {
                return Err(rivet_core::Error::StorageOutOfBounds);
            };
            let pointer = storage_pointer(storage);
            if pointer.is_none() {
                return Err(rivet_core::Error::StorageOutOfBounds);
            }
            Ok(unsafe { pointer.expect("checked above").add(start) })
        })
        .map_err(|err| to_py_err(invalid_argument(err)))
}

fn tensor_array_view_from_pointer<T: numpy::Element>(
    py: Python<'_>,
    tensor: Tensor,
    pointer: *const T,
) -> PyResult<Py<PyAny>> {
    let shape = tensor.dims().to_vec();
    let values = unsafe {
        // The ndarray receives TensorArrayOwner as its base object below, which
        // retains the Tensor and its allocation. Tensor storage never changes
        // allocation size after construction, and the returned ndarray is
        // made read-only before it escapes to Python.
        std::slice::from_raw_parts(pointer, tensor.elem_count())
    };
    let view = ArrayViewD::from_shape(IxDyn(&shape), values)
        .map_err(|err| to_py_err(invalid_argument(err.to_string())))?;
    let owner = Py::new(py, TensorArrayOwner { _tensor: tensor })?;
    let array = unsafe { PyArrayDyn::borrow_from_array(&view, owner.into_bound(py).into_any()) };
    let _ = array.readwrite().make_nonwriteable();
    Ok(array.into_any().unbind())
}

fn tensor_array_copy_to_py(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let shape = tensor.dims().to_vec();
    match tensor.dtype() {
        DType::U8 => Ok(PyArray1::from_vec(
            py,
            tensor
                .to_vec::<u8>()
                .map_err(|err| to_py_err(invalid_argument(err)))?,
        )
        .reshape(shape)?
        .into_any()
        .unbind()),
        DType::F32 => Ok(PyArray1::from_vec(
            py,
            tensor
                .to_vec::<f32>()
                .map_err(|err| to_py_err(invalid_argument(err)))?,
        )
        .reshape(shape)?
        .into_any()
        .unbind()),
        DType::I64 => Ok(PyArray1::from_vec(
            py,
            tensor
                .to_vec::<i64>()
                .map_err(|err| to_py_err(invalid_argument(err)))?,
        )
        .reshape(shape)?
        .into_any()
        .unbind()),
        dtype => Err(to_py_err(invalid_argument(format!(
            "Python tensor conversion does not support {:?}",
            dtype
        )))),
    }
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
