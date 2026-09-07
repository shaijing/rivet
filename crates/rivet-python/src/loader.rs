use crate::dataset::PyArrowDataset;
use crate::error::to_py_err;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rivet_core::batch::ImageBatchBuilder;
use rivet_core::dataset::ArrowImageDataset;
use rivet_core::errors::invalid_argument;
use rivet_core::pipeline::ImagePipeline;
use rivet_core::runtime::ImageDataLoader;
use rivet_core::sample::image::{ImageBatch, ImageBuffer};
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
        self.inner
            .next_batch()
            .map_err(to_py_err)?
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
        .unwrap_or_else(|| ImageBatchBuilder::with_capacity(0).finish());

    image_batch_to_py(py, batch)
}

pub(super) fn image_batch_to_py(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let ImageBatch {
        images,
        labels,
        shape,
        layout,
    } = batch;
    let dtype = images.dtype();
    let images = image_array_to_py(py, images, shape)?;
    let labels = PyArray1::from_vec(py, labels);

    let out = PyDict::new(py);
    out.set_item("images", images)?;
    out.set_item("labels", labels)?;
    out.set_item("shape", shape)?;
    out.set_item("dtype", dtype.as_str())?;
    out.set_item("layout", layout.batch_as_str())?;

    Ok(out.into())
}

fn image_array_to_py(
    py: Python<'_>,
    images: ImageBuffer,
    shape: (usize, usize, usize, usize),
) -> PyResult<Py<PyAny>> {
    match images {
        ImageBuffer::U8(values) => Ok(PyArray1::from_vec(py, values)
            .reshape(shape)?
            .into_any()
            .unbind()),
        ImageBuffer::F32(values) => Ok(PyArray1::from_vec(py, values)
            .reshape(shape)?
            .into_any()
            .unbind()),
    }
}
