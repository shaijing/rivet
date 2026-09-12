use crate::error::to_py_err;
use crate::loader::image_batch_to_py;
use crate::pipeline::PyImagePipeline;
use arrow_buffer::Buffer;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rivet_dataset::batch::ImageBatchBuilder;
use rivet_dataset::dataset::{
    ArrowImageDataset, DEFAULT_ENCODED_CHUNK_SIZE, Dataset, DatasetBundle, DatasetLoadResult,
    ImageFolderDatasetCore, LanceImageDataset, Source, load_lance_image_dataset,
};
use rivet_dataset::image::decode::decode_rgb;
use rivet_dataset::pipeline::ImagePipeline;
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

#[pyclass(name = "_LanceDataset")]
pub(crate) struct PyLanceDataset {
    pub(crate) inner: Source<rivet_dataset::sample::image::EncodedImageSample>,
}

impl PyLanceDataset {
    pub(crate) fn from_inner(
        inner: Source<rivet_dataset::sample::image::EncodedImageSample>,
    ) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyLanceDataset {
    #[new]
    #[pyo3(signature = (path, image_column="image", label_column="label"))]
    fn new(path: PathBuf, image_column: &str, label_column: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Source::new(Arc::new(
                LanceImageDataset::open(path, image_column, label_column).map_err(to_py_err)?,
            )),
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

    /// Materialize compressed image payloads into a shared in-memory cache.
    /// Image decoding remains a pipeline operation.
    #[pyo3(signature = (chunk_size=DEFAULT_ENCODED_CHUNK_SIZE))]
    fn cache_encoded(&self, py: Python<'_>, chunk_size: usize) -> PyResult<Self> {
        let source = self.inner.clone();
        let cached = py
            .detach(|| source.cache_encoded(chunk_size))
            .map_err(to_py_err)?;
        Ok(Self::from_inner(cached))
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::from_source(self.inner.clone()),
        }
    }
}

#[pyclass(name = "_LanceDatasetDict")]
pub(crate) struct PyLanceDatasetDict {
    inner: DatasetBundle<rivet_dataset::sample::image::EncodedImageSample>,
}

#[pymethods]
impl PyLanceDatasetDict {
    #[new]
    #[pyo3(signature = (path, image_column="image", label_column="label"))]
    fn new(path: PathBuf, image_column: &str, label_column: &str) -> PyResult<Self> {
        match load_lance_image_dataset(path, image_column, label_column).map_err(to_py_err)? {
            DatasetLoadResult::Bundle(bundle) => Ok(Self { inner: bundle }),
            DatasetLoadResult::Single(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "a .lance path is a single dataset; use LanceDataset for physical datasets",
            )),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.split(name).is_some()
    }

    fn keys(&self) -> Vec<String> {
        self.inner.split_names().map(str::to_owned).collect()
    }

    fn __getitem__(&self, name: &str) -> PyResult<PyLanceDataset> {
        let Some(dataset) = self.inner.split(name) else {
            let available = self.inner.split_names().collect::<Vec<_>>().join(", ");
            return Err(PyKeyError::new_err(format!(
                "split '{name}' not found; available splits: {available}"
            )));
        };
        Ok(PyLanceDataset::from_inner(dataset))
    }
}

#[pyfunction]
#[pyo3(signature = (path, image_column="image", label_column="label", split=None))]
pub(crate) fn load_lance_split(
    path: PathBuf,
    image_column: &str,
    label_column: &str,
    split: Option<&str>,
) -> PyResult<PyLanceDataset> {
    match load_lance_image_dataset(path, image_column, label_column).map_err(to_py_err)? {
        DatasetLoadResult::Single(dataset) => {
            if split.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "split cannot be specified when loading a single .lance dataset path",
                ));
            }
            Ok(PyLanceDataset::from_inner(dataset))
        }
        DatasetLoadResult::Bundle(bundle) => {
            let split = split.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "dataset root contains multiple splits; specify split",
                )
            })?;
            let Some(dataset) = bundle.split(split) else {
                let available = bundle.split_names().collect::<Vec<_>>().join(", ");
                return Err(PyKeyError::new_err(format!(
                    "split '{split}' not found; available splits: {available}"
                )));
            };
            Ok(PyLanceDataset::from_inner(dataset))
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
    sample: rivet_dataset::sample::image::EncodedImageSample,
) -> PyResult<Py<PyDict>> {
    let decoded = decode_rgb(sample.image.as_slice(), sample.label).map_err(to_py_err)?;
    let mut batch = ImageBatchBuilder::with_capacity(1);
    batch.push(decoded).map_err(to_py_err)?;
    image_batch_to_py(py, batch.finish())
}
