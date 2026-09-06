use crate::batch::ImageBatchBuilder;
use crate::dataset::ArrowImageDatasetCore;
use crate::errors::value_err;
use crate::pipeline::ImagePipeline;
use crate::python::dataset::PyArrowDataset;
use crate::runtime::ImageDataLoader;
use crate::sample::{ImageBatch, ImageBuffer};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
                .batch(batch_size, false)?
                .compile()?,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .next_batch()?
            .map(|batch| image_batch_to_py(py, batch))
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
        return Err(value_err("arrow_files must not be empty"));
    }

    let dataset = Arc::new(ArrowImageDatasetCore::new(
        arrow_files,
        image_column.to_string(),
        label_column.to_string(),
    )?);
    let mut loader = ImagePipeline::new(dataset)
        .decode_image()
        .batch(batch_size, false)?
        .compile_from(start)?;
    let batch = loader
        .next_batch()?
        .unwrap_or_else(|| ImageBatchBuilder::with_capacity(0).finish());

    image_batch_to_py(py, batch)
}

pub(super) fn image_batch_to_py(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    let _image_nbytes = batch.images.byte_len();
    out.set_item("images", image_array_to_py(py, &batch)?)?;
    out.set_item("labels", PyArray1::from_vec(py, batch.labels))?;
    out.set_item("shape", batch.shape)?;
    out.set_item("dtype", batch.images.dtype().as_str())?;
    out.set_item("layout", batch.layout.batch_as_str())?;

    Ok(out.into())
}

fn image_array_to_py(py: Python<'_>, batch: &ImageBatch) -> PyResult<Py<PyAny>> {
    match &batch.images {
        ImageBuffer::U8(values) => Ok(PyArray1::from_vec(py, values.clone())
            .reshape(batch.shape)?
            .into_any()
            .unbind()),
        ImageBuffer::F32(values) => Ok(PyArray1::from_vec(py, values.clone())
            .reshape(batch.shape)?
            .into_any()
            .unbind()),
    }
}
