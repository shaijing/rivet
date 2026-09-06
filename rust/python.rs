use crate::arrow_dataset::ArrowImageDatasetCore;
use crate::batch::ImageBatchBuilder;
use crate::dataset::Dataset;
use crate::decoder::decode_rgb;
use crate::errors::value_err;
use crate::pipeline::ImagePipeline;
use crate::sample::ImageBatch;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass(name = "_ArrowDataset")]
pub(crate) struct PyArrowDataset {
    pub(crate) inner: Arc<ArrowImageDatasetCore>,
}

#[pymethods]
impl PyArrowDataset {
    #[new]
    #[pyo3(signature = (arrow_files, image_column="img", label_column="label"))]
    fn new(arrow_files: Vec<PathBuf>, image_column: &str, label_column: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(ArrowImageDatasetCore::new(
                arrow_files,
                image_column.to_string(),
                label_column.to_string(),
            )?),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn get_encoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = self.inner.get(index)?;
        let out = PyDict::new(py);
        out.set_item("image", PyBytes::new(py, &sample.image))?;
        out.set_item("label", sample.label)?;
        Ok(out.into())
    }

    fn get_decoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = self.inner.get(index)?;
        let decoded = decode_rgb(&sample.image, sample.label)?;
        let mut batch = ImageBatchBuilder::with_capacity(1);
        batch.push(decoded)?;
        image_batch_to_py(py, batch.finish())
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::new(Arc::clone(&self.inner)),
        }
    }
}

#[pyclass(name = "_ImagePipeline")]
pub(crate) struct PyImagePipeline {
    pub(crate) inner: ImagePipeline,
}

#[pymethods]
impl PyImagePipeline {
    fn decode_image(&self) -> Self {
        Self {
            inner: self.inner.clone().decode_image(),
        }
    }

    fn batch(&self, size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().batch(size)?,
        })
    }

    fn execute(&self) -> PyResult<PyDataLoader> {
        Ok(PyDataLoader {
            inner: self.inner.clone().compile()?,
        })
    }
}

#[pyclass(name = "_DataLoader")]
pub(crate) struct PyDataLoader {
    inner: crate::dataloader::ImageDataLoader,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    fn new(dataset: &PyArrowDataset, batch_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: ImagePipeline::new(Arc::clone(&dataset.inner))
                .decode_image()
                .batch(batch_size)?
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
        .batch(batch_size)?
        .compile_from(start)?;
    let batch = loader
        .next_batch()?
        .unwrap_or_else(|| ImageBatchBuilder::with_capacity(0).finish());

    image_batch_to_py(py, batch)
}

fn image_batch_to_py(py: Python<'_>, batch: ImageBatch) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("images", PyBytes::new(py, &batch.images))?;
    out.set_item("labels", batch.labels)?;
    out.set_item("shape", batch.shape)?;
    out.set_item("dtype", "uint8")?;
    out.set_item("layout", "NHWC")?;

    Ok(out.into())
}
