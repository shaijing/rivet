use crate::error::to_py_err;
use crate::loader::image_batch_to_py;
use crate::pipeline::PyImagePipeline;
use arrow_buffer::Buffer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rivet_core::batch::ImageBatchBuilder;
use rivet_core::dataset::{ArrowImageDataset, Dataset, ImageFolderDatasetCore};
use rivet_core::image::decode::decode_rgb;
use rivet_core::pipeline::ImagePipeline;
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass(name = "_ArrowDataset")]
pub(crate) struct PyArrowDataset {
    pub(crate) inner: Arc<ArrowImageDataset>,
}

#[pymethods]
impl PyArrowDataset {
    #[new]
    #[pyo3(signature = (arrow_files, image_column="img", label_column="label"))]
    fn new(arrow_files: Vec<PathBuf>, image_column: &str, label_column: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(
                ArrowImageDataset::new(
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
        encoded_sample_to_py(py, sample.image, sample.label)
    }

    fn get_decoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        decoded_sample_to_py(py, self.inner.get(index).map_err(to_py_err)?)
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::new(Arc::clone(&self.inner)),
        }
    }
}

#[pyclass(name = "_ImageFolderDataset")]
pub(crate) struct PyImageFolderDataset {
    pub(crate) inner: Arc<ImageFolderDatasetCore>,
}

#[pymethods]
impl PyImageFolderDataset {
    #[new]
    fn new(root: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(ImageFolderDatasetCore::new(root).map_err(to_py_err)?),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn classes(&self) -> Vec<String> {
        self.inner.classes().to_vec()
    }

    #[getter]
    fn class_to_idx(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        for (class, index) in self.inner.class_to_idx() {
            out.set_item(class, index)?;
        }
        Ok(out.into())
    }

    #[getter]
    fn samples(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let out = PyList::empty(py);
        for sample in self.inner.samples() {
            let item = PyDict::new(py);
            item.set_item("path", sample.path.to_string_lossy().as_ref())?;
            item.set_item("label", sample.label)?;
            out.append(item)?;
        }
        Ok(out.into())
    }

    fn get_encoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = self.inner.get(index).map_err(to_py_err)?;
        encoded_sample_to_py(py, sample.image, sample.label)
    }

    fn get_decoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        decoded_sample_to_py(py, self.inner.get(index).map_err(to_py_err)?)
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::new(Arc::clone(&self.inner)),
        }
    }
}

fn encoded_sample_to_py(py: Python<'_>, image: Buffer, label: i64) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    // PyBytes::new copies the payload; this is the Python API boundary, not
    // the training runtime hot path.
    out.set_item("image", PyBytes::new(py, image.as_slice()))?;
    out.set_item("label", label)?;
    Ok(out.into())
}

fn decoded_sample_to_py(
    py: Python<'_>,
    sample: rivet_core::sample::image::EncodedImageSample,
) -> PyResult<Py<PyDict>> {
    let decoded = decode_rgb(sample.image.as_slice(), sample.label).map_err(to_py_err)?;
    let mut batch = ImageBatchBuilder::with_capacity(1);
    batch.push(decoded).map_err(to_py_err)?;
    image_batch_to_py(py, batch.finish())
}
