use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyMapping, PyString};
use rivet_data::dataset::{self, DataFiles, DownloadFormat, DownloadOptions, TorchDatasetProvider};
use rivet_data::{DataError, DataResult};
use rivet_vision::VisionError;

use crate::error::to_py_err;

struct Torchvision {
    error: RefCell<Option<PyErr>>,
}

impl Torchvision {
    fn call(
        &self,
        function: &str,
        location: &str,
        root: &Path,
        output: Option<&Path>,
    ) -> DataResult<()> {
        Python::attach(|py| {
            let module = py.import("rivet._torch_dataset")?;
            match output {
                Some(output) => module.call_method1(function, (location, root, output))?,
                None => module.call_method1(function, (location, root))?,
            };
            Ok(())
        })
        .map_err(|err| {
            self.error.replace(Some(err));
            DataError::Io(std::io::Error::other(
                "torchvision dataset operation failed",
            ))
        })
    }
}

impl TorchDatasetProvider for Torchvision {
    fn download(&self, location: &str, destination: &Path) -> DataResult<()> {
        self.call("_download_torch", location, destination, None)
    }

    fn convert(&self, location: &str, root: &Path, destination: &Path) -> DataResult<()> {
        self.call("_convert_torch", location, root, Some(destination))
    }
}

/// Return the local path to a downloaded dataset or derived Lance bundle.
///
/// Set offline=True to reuse only completed cache entries and extract=False
/// to keep HTTP archives. Use data_files and json_columns with format="lance".
#[pyfunction]
#[pyo3(signature = (source, *, revision=None, cache_dir=None, offline=false, extract=true, format="raw", data_files=None, json_columns=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn download_dataset(
    py: Python<'_>,
    source: String,
    revision: Option<String>,
    cache_dir: Option<PathBuf>,
    offline: bool,
    extract: bool,
    format: &str,
    data_files: Option<Bound<'_, PyAny>>,
    json_columns: Option<Bound<'_, PyAny>>,
) -> PyResult<PathBuf> {
    let format = match format {
        "raw" => DownloadFormat::Raw,
        "lance" => DownloadFormat::Lance,
        _ => return Err(PyValueError::new_err("format must be 'raw' or 'lance'")),
    };
    if format == DownloadFormat::Raw && (data_files.is_some() || json_columns.is_some()) {
        return Err(PyValueError::new_err(
            "data_files and json_columns require format='lance'",
        ));
    }
    let data_files = data_files.as_ref().map(extract_data_files).transpose()?;
    let json_columns = json_columns
        .as_ref()
        .map(|columns| {
            if columns.is_instance_of::<PyString>() {
                return Err(PyValueError::new_err(
                    "json_columns must contain unique, nonempty column names",
                ));
            }
            columns.extract::<Vec<String>>().map_err(|_| {
                PyValueError::new_err("json_columns must contain unique, nonempty column names")
            })
        })
        .transpose()?;
    let cache_dir = expand_cache_dir(py, cache_dir)?;
    let options = DownloadOptions {
        revision,
        cache_dir,
        offline,
        keep_archive: !extract,
        format,
        data_files,
        json_columns,
    };
    with_torchvision(py, move |provider| {
        Ok(dataset::download_dataset_with_torch(
            &source,
            &options,
            Some(provider),
        )?)
    })
}

pub(crate) fn expand_cache_dir(
    py: Python<'_>,
    cache_dir: Option<PathBuf>,
) -> PyResult<Option<PathBuf>> {
    cache_dir
        .map(|path| -> PyResult<PathBuf> {
            if path.to_string_lossy().starts_with('~') {
                py.import("pathlib")?
                    .getattr("Path")?
                    .call1((path,))?
                    .call_method0("expanduser")?
                    .extract()
            } else {
                Ok(path)
            }
        })
        .transpose()
}

pub(crate) fn with_torchvision<T: Send>(
    py: Python<'_>,
    operation: impl FnOnce(&dyn TorchDatasetProvider) -> Result<T, VisionError> + Send,
) -> PyResult<T> {
    py.detach(move || {
        let provider = Torchvision {
            error: RefCell::new(None),
        };
        operation(&provider).map_err(|err| {
            provider.error.into_inner().unwrap_or_else(|| match err {
                VisionError::Data(DataError::Io(err)) if err.kind() == ErrorKind::NotFound => {
                    PyFileNotFoundError::new_err(err.to_string())
                }
                VisionError::Data(DataError::Io(err)) if err.kind() == ErrorKind::Unsupported => {
                    PyRuntimeError::new_err(err.to_string())
                }
                err => to_py_err(err),
            })
        })
    })
}

fn extract_data_files(value: &Bound<'_, PyAny>) -> PyResult<DataFiles> {
    if value.is_instance_of::<PyString>() {
        return Ok(DataFiles::File(value.extract()?));
    }
    let mapping = value
        .cast::<PyMapping>()
        .map_err(|_| PyTypeError::new_err("data_files must be a string or a mapping of splits"))?;
    let mut splits = BTreeMap::new();
    for item in mapping.items()?.try_iter()? {
        let (split, patterns): (String, Bound<'_, PyAny>) = item?.extract()?;
        let patterns = if patterns.is_instance_of::<PyString>() {
            vec![patterns.extract()?]
        } else {
            patterns.extract()?
        };
        splits.insert(split, patterns);
    }
    Ok(DataFiles::Splits(splits))
}

#[pyfunction]
pub(crate) fn _dataset_cache() -> PyResult<PathBuf> {
    dataset::dataset_cache_dir().map_err(to_py_err)
}
