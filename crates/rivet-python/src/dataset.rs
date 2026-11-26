use crate::error::to_py_err;
use crate::loader::image_batch_to_py;
use crate::pipeline::PyImagePipeline;
use arrow_buffer::Buffer;
#[cfg(feature = "lance")]
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rivet_data::dataset::Dataset;
#[cfg(feature = "lance")]
use rivet_data::dataset::{DatasetBundle, DatasetLoadResult};
use rivet_vision::api::{
    ArrowImageDataset, ImageFolderDatasetCore, ImagePipeline, decode_image_batch,
};
#[cfg(feature = "lance")]
use rivet_vision::api::{
    CacheConfig, CacheLevel, DEFAULT_ENCODED_CHUNK_SIZE, DatasetLocation, ImageSource,
    LanceImageDataset, LoadDatasetOptions, load_dataset_with_torch, load_lance_image_dataset,
};
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
        let sample = self.inner.get(index).map_err(to_py_err)?;
        image_batch_to_py(
            py,
            decode_image_batch(sample.image.as_slice(), sample.label).map_err(to_py_err)?,
        )
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::new(Arc::clone(&self.inner)),
        }
    }
}

#[cfg(feature = "lance")]
#[pyclass(name = "_LanceDataset")]
pub(crate) struct PyLanceDataset {
    pub(crate) inner: ImageSource,
}

#[cfg(feature = "lance")]
impl PyLanceDataset {
    pub(crate) fn from_inner(inner: ImageSource) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "lance")]
#[pymethods]
impl PyLanceDataset {
    #[new]
    #[pyo3(signature = (path, image_column="image", label_column="label"))]
    fn new(path: PathBuf, image_column: &str, label_column: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ImageSource::from_encoded(Arc::new(
                LanceImageDataset::open(path, image_column, label_column).map_err(to_py_err)?,
            )),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn path(&self) -> Option<PathBuf> {
        match &self.inner {
            ImageSource::Encoded(source) => {
                source.as_dataset().source_path().map(ToOwned::to_owned)
            }
            _ => None,
        }
    }

    fn get_encoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        let sample = self.inner.get_encoded(index).map_err(to_py_err)?;
        encoded_sample_to_py(py, sample.image, sample.label)
    }

    fn get_decoded(&self, py: Python<'_>, index: usize) -> PyResult<Py<PyDict>> {
        image_batch_to_py(py, self.inner.get_decoded_batch(index).map_err(to_py_err)?)
    }

    /// Materialize encoded or decoded image payloads into a shared in-memory
    /// cache. Decoded caching moves the pipeline boundary past image decode.
    #[pyo3(signature = (level="encoded", chunk_size=DEFAULT_ENCODED_CHUNK_SIZE, max_bytes=None))]
    fn cache(
        &self,
        py: Python<'_>,
        level: &str,
        chunk_size: usize,
        max_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let policy = match level {
            "encoded" => {
                if max_bytes.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "max_bytes is only supported for decoded caching",
                    ));
                }
                CacheConfig {
                    level: CacheLevel::Encoded,
                    chunk_size,
                    max_bytes: None,
                }
            }
            "decoded" => CacheConfig {
                level: CacheLevel::Decoded,
                chunk_size,
                max_bytes,
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cache level must be 'encoded' or 'decoded'",
                ));
            }
        };
        let source = self.inner.clone();
        let cached = py
            .detach(|| source.cache_config(policy))
            .map_err(to_py_err)?;
        Ok(Self::from_inner(cached))
    }

    fn pipeline(&self) -> PyImagePipeline {
        PyImagePipeline {
            inner: ImagePipeline::from_source(self.inner.clone()),
        }
    }
}

#[cfg(feature = "lance")]
#[pyclass(name = "_LanceDatasetDict")]
pub(crate) struct PyLanceDatasetDict {
    inner: DatasetBundle<ImageSource>,
}

#[cfg(feature = "lance")]
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

/// Open a local or remote Lance image dataset, optionally selecting one split.
#[pyfunction]
#[pyo3(signature = (path, *, split=None, revision=None, cache_dir=None, offline=false, image_column="image", label_column="label"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_dataset<'py>(
    py: Python<'py>,
    path: Bound<'py, PyAny>,
    split: Option<String>,
    revision: Option<String>,
    cache_dir: Option<PathBuf>,
    offline: bool,
    image_column: &str,
    label_column: &str,
) -> PyResult<Bound<'py, PyAny>> {
    #[cfg(not(feature = "lance"))]
    {
        let _ = (
            py,
            path,
            split,
            revision,
            cache_dir,
            offline,
            image_column,
            label_column,
        );
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Lance support is disabled; rebuild rivet-python with the 'lance' feature",
        ))
    }

    #[cfg(feature = "lance")]
    {
        use crate::download::{expand_cache_dir, with_torchvision};
        use pyo3::types::{PyString, PyStringMethods};
        use rivet_data::dataset::is_remote_dataset;

        let remote = if let Ok(source) = path.cast::<PyString>() {
            source
                .to_cow()
                .ok()
                .filter(|source| is_remote_dataset(source))
                .map(|source| source.into_owned())
        } else {
            None
        };
        let cache_dir = if remote.is_some() {
            expand_cache_dir(py, cache_dir)?
        } else {
            None
        };
        let local = if remote.is_none() {
            Some(path.extract::<PathBuf>()?)
        } else {
            None
        };
        let options = LoadDatasetOptions {
            split,
            revision,
            cache_dir,
            offline,
            image_column: image_column.to_owned(),
            label_column: label_column.to_owned(),
        };
        let loaded = with_torchvision(py, move |provider| {
            let location = match &remote {
                Some(source) => DatasetLocation::Remote(source),
                None => DatasetLocation::Local(local.as_deref().unwrap()),
            };
            load_dataset_with_torch(location, &options, Some(provider))
        })?;
        let wrappers = py.import("rivet._datasets")?;
        match loaded {
            DatasetLoadResult::Single(dataset) => wrappers.getattr("LanceDataset")?.call_method1(
                "_from_inner",
                (Py::new(py, PyLanceDataset::from_inner(dataset))?,),
            ),
            DatasetLoadResult::Bundle(inner) => wrappers
                .getattr("DatasetDict")?
                .call1((Py::new(py, PyLanceDatasetDict { inner })?,)),
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
        let sample = self.inner.get(index).map_err(to_py_err)?;
        image_batch_to_py(
            py,
            decode_image_batch(sample.image.as_slice(), sample.label).map_err(to_py_err)?,
        )
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

#[cfg(feature = "lance")]
#[pyfunction]
#[pyo3(signature = (root, output, *, split=None, overwrite=false))]
pub(crate) fn _convert_imagefolder_lance(
    py: Python<'_>,
    root: PathBuf,
    output: PathBuf,
    split: Option<String>,
    overwrite: bool,
) -> PyResult<PathBuf> {
    py.detach(move || {
        rivet_vision::api::convert_imagefolder_dataset(root, output, split.as_deref(), overwrite)
            .map(|report| report.output_root)
    })
    .map_err(to_py_err)
}

#[cfg(feature = "lance")]
#[pyfunction]
pub(crate) fn _convert_torch_images(
    py: Python<'_>,
    output: PathBuf,
    splits: std::collections::BTreeMap<String, rivet_vision::datasets::lance::TorchImageSplit>,
    classes: Option<Vec<String>>,
) -> PyResult<()> {
    py.detach(move || {
        rivet_vision::datasets::lance::convert_torch_images(&output, splits, classes.as_deref())
    })
    .map_err(to_py_err)
}

#[pyfunction]
pub(crate) fn _list_datasets(py: Python<'_>, cache_dir: PathBuf) -> PyResult<Vec<Py<PyDict>>> {
    let entries = py
        .detach(move || rivet_data::dataset::list_datasets(cache_dir))
        .map_err(to_py_err)?;
    entries
        .into_iter()
        .map(|entry| {
            let result = PyDict::new(py);
            result.set_item("name", entry.name)?;
            result.set_item("path", entry.path)?;
            result.set_item("kind", entry.kind)?;
            result.set_item("provider", entry.provider)?;
            result.set_item("source", entry.source)?;
            result.set_item("revision", entry.revision)?;
            result.set_item("splits", entry.splits)?;
            Ok(result.unbind())
        })
        .collect()
}
