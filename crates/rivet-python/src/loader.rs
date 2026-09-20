use crate::dataset::PyArrowDataset;
use crate::dlpack::PyDLPackTensor;
use crate::dtype;
use crate::error::to_py_err;
use crate::numpy::tensor_to_numpy;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
    let ImageBatch {
        images,
        labels,
        axis_order,
    } = batch;
    let dtype = images.dtype();
    let shape = images.dims().to_vec();
    let images = tensor_to_numpy(py, images)?;
    let labels = tensor_to_numpy(py, labels)?;

    let out = PyDict::new(py);
    out.set_item("images", images)?;
    out.set_item("labels", labels)?;
    out.set_item("shape", &shape)?;
    out.set_item("dtype", dtype::name(dtype))?;
    out.set_item("layout", axis_order.batch_as_str())?;

    Ok(out.into())
}

fn image_batch_to_dlpack(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let ImageBatch {
        images,
        labels,
        axis_order,
    } = batch;
    let shape = images.dims().to_vec();
    let dtype_name = dtype::name(images.dtype());
    let out = PyDict::new(py);
    out.set_item("images", Py::new(py, PyDLPackTensor::new(images))?)?;
    out.set_item("labels", Py::new(py, PyDLPackTensor::new(labels))?)?;
    out.set_item("shape", shape)?;
    out.set_item("dtype", dtype_name)?;
    out.set_item("layout", axis_order.batch_as_str())?;
    Ok(out.into())
}
