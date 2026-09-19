use crate::dataset::PyArrowDataset;
use crate::error::to_py_err;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rivet_core::{DType, Tensor};
use rivet_vision::batch::ImageBatchBuilder;
use rivet_vision::datasets::ArrowImageDataset;
use rivet_vision::errors::invalid_argument;
use rivet_vision::pipeline::ImagePipeline;
use rivet_vision::runtime::ImageDataLoader;
use rivet_vision::sample::image::ImageBatch;
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
    let batch = loader.next_batch().map_err(to_py_err)?.unwrap_or_else(|| {
        ImageBatchBuilder::with_capacity(0)
            .finish()
            .expect("empty batch")
    });

    image_batch_to_py(py, batch)
}

pub(super) fn image_batch_to_py(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let ImageBatch { images, labels } = batch;
    let dtype = images.dtype();
    let shape = images.dims().to_vec();
    let images = image_array_to_py(py, images)?;
    let labels = PyArray1::from_vec(
        py,
        labels
            .to_vec::<i64>()
            .map_err(|err| to_py_err(invalid_argument(err)))?,
    );

    let out = PyDict::new(py);
    out.set_item("images", images)?;
    out.set_item("labels", labels)?;
    out.set_item("shape", &shape)?;
    out.set_item("dtype", dtype_name(dtype))?;
    out.set_item("layout", batch_layout(&shape))?;

    Ok(out.into())
}

fn image_array_to_py(py: Python<'_>, images: Tensor) -> PyResult<Py<PyAny>> {
    let shape = images.dims().to_vec();
    match images.dtype() {
        DType::U8 => Ok(PyArray1::from_vec(
            py,
            images
                .to_vec::<u8>()
                .map_err(|err| to_py_err(invalid_argument(err)))?,
        )
        .reshape(shape)?
        .into_any()
        .unbind()),
        DType::F32 => Ok(PyArray1::from_vec(
            py,
            images
                .to_vec::<f32>()
                .map_err(|err| to_py_err(invalid_argument(err)))?,
        )
        .reshape(shape)?
        .into_any()
        .unbind()),
        dtype => Err(to_py_err(invalid_argument(format!(
            "Python image conversion does not support {:?}",
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
