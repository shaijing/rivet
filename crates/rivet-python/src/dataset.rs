use crate::error::to_py_err;
use crate::loader::image_batch_to_py;
use crate::pipeline::PyImagePipeline;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rivet_core::batch::ImageBatchBuilder;
use rivet_core::dataset::{ArrowImageDatasetCore, Dataset};
use rivet_core::image::decode::decode_rgb;
use rivet_core::pipeline::ImagePipeline;
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
            inner: Arc::new(
                ArrowImageDatasetCore::new(
                    arrow_files,
                    image_column.to_string(),
                    label_column.to_string(),
                )
                .map_err(to_py_err)?,
            ),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn get_encoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = self.inner.get(index).map_err(to_py_err)?;
        let out = PyDict::new(py);
        out.set_item("image", PyBytes::new(py, &sample.image))?;
        out.set_item("label", sample.label)?;
        Ok(out.into())
    }

    fn get_decoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = self.inner.get(index).map_err(to_py_err)?;
        let decoded = decode_rgb(&sample.image, sample.label).map_err(to_py_err)?;
        let mut batch = ImageBatchBuilder::with_capacity(1);
        batch.push(decoded).map_err(to_py_err)?;
        image_batch_to_py(py, batch.finish())
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::new(Arc::clone(&self.inner)),
        }
    }
}
