use crate::batch::ImageBatchBuilder;
use crate::dataset::{ArrowImageDatasetCore, Dataset};
use crate::image::decode::decode_rgb;
use crate::pipeline::ImagePipeline;
use crate::python::loader::image_batch_to_py;
use crate::python::pipeline::PyImagePipeline;
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
