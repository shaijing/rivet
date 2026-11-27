use crate::numpy::tensor_to_numpy;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rivet_data::dataset::Dataset;
use rivet_text::{
    ArrowTextDataset, EncodeOptions, LabelMode, PaddingDirection, PaddingParams, PaddingStrategy,
    TextColumns, TextDataLoader, TextEncodingErrors, TextError, TextFileDataset, TextFileOptions,
    TextPipeline, TextRecordsDataset, TextRecordsOptions, TextSample, TextSampleBy, TextTokenizer,
};
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTextDataset>()?;
    module.add_class::<PyTextTokenizer>()?;
    module.add_class::<PyTextPipeline>()?;
    module.add_class::<PyTextDataLoader>()
}

fn to_py_err(error: TextError) -> PyErr {
    match error {
        TextError::Io(error) => error.into(),
        error => PyValueError::new_err(error.to_string()),
    }
}

#[derive(FromPyObject)]
enum TextPaths {
    One(PathBuf),
    Many(Vec<PathBuf>),
}

impl TextPaths {
    fn into_paths(self) -> Vec<PathBuf> {
        match self {
            Self::One(path) => vec![path],
            Self::Many(paths) => paths,
        }
    }
}

fn file_options(
    sample_by: &str,
    encoding: &str,
    errors: &str,
    keep_linebreaks: bool,
) -> PyResult<TextFileOptions> {
    let sample_by = match sample_by {
        "line" => TextSampleBy::Line,
        "paragraph" => TextSampleBy::Paragraph,
        "document" => TextSampleBy::Document,
        _ => {
            return Err(PyValueError::new_err(
                "sample_by must be line, paragraph, or document",
            ));
        }
    };
    let encoding_errors = match errors {
        "strict" => TextEncodingErrors::Strict,
        "replace" => TextEncodingErrors::Replace,
        "ignore" => TextEncodingErrors::Ignore,
        _ => {
            return Err(PyValueError::new_err(
                "errors must be strict, replace, or ignore",
            ));
        }
    };
    Ok(TextFileOptions {
        sample_by,
        keep_linebreaks,
        encoding: encoding.parse().map_err(to_py_err)?,
        encoding_errors,
    })
}

#[pyclass(name = "TextDataset", module = "rivet.text", frozen)]
struct PyTextDataset {
    inner: Arc<dyn Dataset<Item = TextSample>>,
}

#[pymethods]
impl PyTextDataset {
    /// Read plain-text files in caller order. Accepts one path or a sequence of paths.
    #[staticmethod]
    #[pyo3(signature = (paths, *, sample_by="line", encoding="utf-8", errors="strict", keep_linebreaks=false))]
    fn from_files(
        py: Python<'_>,
        paths: TextPaths,
        sample_by: &str,
        encoding: &str,
        errors: &str,
        keep_linebreaks: bool,
    ) -> PyResult<Self> {
        let options = file_options(sample_by, encoding, errors, keep_linebreaks)?;
        let source = py
            .detach(|| TextFileDataset::open_files(paths.into_paths(), options))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(source),
        })
    }

    /// Read JSON, JSONL, CSV, TSV, spreadsheets, TAR archives, or TXT files.
    #[staticmethod]
    #[pyo3(signature = (paths, *, text_column="text", text_pair_column=None, label_column=None, sheet=None, sample_by="line", encoding="utf-8", errors="strict", keep_linebreaks=false))]
    #[allow(clippy::too_many_arguments)]
    fn from_records(
        py: Python<'_>,
        paths: TextPaths,
        text_column: &str,
        text_pair_column: Option<String>,
        label_column: Option<String>,
        sheet: Option<String>,
        sample_by: &str,
        encoding: &str,
        errors: &str,
        keep_linebreaks: bool,
    ) -> PyResult<Self> {
        let options = TextRecordsOptions {
            columns: TextColumns {
                text: text_column.into(),
                text_pair: text_pair_column,
                label: label_column,
            },
            sheet,
            text: file_options(sample_by, encoding, errors, keep_linebreaks)?,
        };
        let source = py
            .detach(|| TextRecordsDataset::open_files(paths.into_paths(), options))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(source),
        })
    }

    /// Read Arrow IPC File or Stream from a local path.
    #[staticmethod]
    #[pyo3(signature = (path, *, text_column="text", text_pair_column=None, label_column=None))]
    fn from_arrow(
        py: Python<'_>,
        path: PathBuf,
        text_column: &str,
        text_pair_column: Option<String>,
        label_column: Option<String>,
    ) -> PyResult<Self> {
        let columns = TextColumns {
            text: text_column.into(),
            text_pair: text_pair_column,
            label: label_column,
        };
        let source = py
            .detach(|| ArrowTextDataset::open_ipc(path, columns))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Arc::new(source),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return a (text, text_pair, label) tuple. Missing fields are None.
    fn __getitem__(
        &self,
        py: Python<'_>,
        index: isize,
    ) -> PyResult<(String, Option<String>, Option<i64>)> {
        let len = self.inner.len();
        let index = if index < 0 {
            len.checked_sub(index.unsigned_abs())
        } else {
            Some(index as usize)
        };
        let index = index
            .filter(|&index| index < len)
            .ok_or_else(|| PyIndexError::new_err("text dataset index out of range"))?;
        let row = py
            .detach(|| self.inner.get(index))
            .map_err(|error| to_py_err(error.into()))?;
        Ok((row.text, row.text_pair, row.label))
    }
}

#[pyclass(name = "TextTokenizer", module = "rivet.text", frozen)]
struct PyTextTokenizer {
    inner: TextTokenizer,
}

#[pymethods]
impl PyTextTokenizer {
    #[staticmethod]
    fn from_file(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| TextTokenizer::from_file(path))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| TextTokenizer::from_json(json))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    #[pyo3(signature = (text, text_pair=None, *, add_special_tokens=true))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        text_pair: Option<&str>,
        add_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut options = EncodeOptions::default();
        options.add_special_tokens = add_special_tokens;
        let encoded = py
            .detach(|| match text_pair {
                Some(pair) => self.inner.encode_pair(text, pair, &options),
                None => self.inner.encode(text, &options),
            })
            .map_err(to_py_err)?;
        let fields = PyDict::new(py);
        fields.set_item("input_ids", encoded.input_ids)?;
        fields.set_item(
            "attention_mask",
            encoded
                .attention_mask
                .into_iter()
                .map(u32::from)
                .collect::<Vec<_>>(),
        )?;
        fields.set_item(
            "token_type_ids",
            encoded
                .token_type_ids
                .into_iter()
                .map(u32::from)
                .collect::<Vec<_>>(),
        )?;
        Ok(fields)
    }

    #[pyo3(signature = (ids, *, skip_special_tokens=true))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        py.detach(|| self.inner.decode(&ids, skip_special_tokens))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "TextPipeline", module = "rivet.text", frozen)]
struct PyTextPipeline {
    inner: TextPipeline,
}

#[pymethods]
impl PyTextPipeline {
    #[new]
    fn new(dataset: &PyTextDataset, tokenizer: &PyTextTokenizer) -> Self {
        Self {
            inner: TextPipeline::new(Arc::clone(&dataset.inner), tokenizer.inner.clone()),
        }
    }

    #[pyo3(signature = (size, *, drop_last=false))]
    fn batch(&self, size: usize, drop_last: bool) -> Self {
        Self {
            inner: self.inner.clone().batch(size, drop_last),
        }
    }

    fn truncate(&self, max_length: usize) -> Self {
        Self {
            inner: self.inner.clone().truncate(max_length),
        }
    }

    fn add_special_tokens(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().add_special_tokens(enabled),
        }
    }

    fn no_truncation(&self) -> Self {
        Self {
            inner: self.inner.clone().no_truncation(),
        }
    }

    /// Pad to the longest sequence in each batch, or to length when supplied.
    #[pyo3(signature = (pad_id, *, pad_token="[PAD]", length=None, side="right", pad_to_multiple_of=None))]
    fn padding(
        &self,
        pad_id: u32,
        pad_token: &str,
        length: Option<usize>,
        side: &str,
        pad_to_multiple_of: Option<usize>,
    ) -> PyResult<Self> {
        let direction = match side {
            "left" => PaddingDirection::Left,
            "right" => PaddingDirection::Right,
            _ => return Err(PyValueError::new_err("side must be left or right")),
        };
        let mut params = PaddingParams::default();
        params.strategy = length.map_or(PaddingStrategy::BatchLongest, PaddingStrategy::Fixed);
        params.direction = direction;
        params.pad_id = pad_id;
        params.pad_token = pad_token.into();
        params.pad_to_multiple_of = pad_to_multiple_of;
        Ok(Self {
            inner: self.inner.clone().padding(params),
        })
    }

    fn no_padding(&self) -> Self {
        Self {
            inner: self.inner.clone().no_padding(),
        }
    }

    fn indices(&self, indices: Vec<usize>) -> Self {
        Self {
            inner: self.inner.clone().indices(indices),
        }
    }

    fn shuffle(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().shuffle(seed),
        }
    }

    fn epoch(&self, epoch: u64) -> Self {
        Self {
            inner: self.inner.clone().epoch(epoch),
        }
    }

    /// Choose per-sample labels, no labels, or unshifted causal language modeling labels.
    #[pyo3(signature = (mode, *, ignore_index=-100))]
    fn labels(&self, mode: &str, ignore_index: i64) -> PyResult<Self> {
        let mode = match mode {
            "sample" => LabelMode::Sample,
            "none" => LabelMode::None,
            "causal_lm" => LabelMode::CausalLm { ignore_index },
            _ => {
                return Err(PyValueError::new_err(
                    "mode must be sample, none, or causal_lm",
                ));
            }
        };
        Ok(Self {
            inner: self.inner.clone().labels(mode),
        })
    }

    fn compile(&self, py: Python<'_>) -> PyResult<PyTextDataLoader> {
        let inner = py
            .detach(|| self.inner.clone().compile())
            .map_err(to_py_err)?;
        Ok(PyTextDataLoader { inner })
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyTextDataLoader> {
        self.compile(py)
    }
}

#[pyclass(name = "TextDataLoader", module = "rivet.text")]
struct PyTextDataLoader {
    inner: TextDataLoader,
}

#[pymethods]
impl PyTextDataLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(batch) = py.detach(|| self.inner.next_batch()).map_err(to_py_err)? else {
            return Ok(None);
        };
        let fields = PyDict::new(py);
        fields.set_item("input_ids", tensor_to_numpy(py, batch.input_ids)?)?;
        fields.set_item("attention_mask", tensor_to_numpy(py, batch.attention_mask)?)?;
        fields.set_item("token_type_ids", tensor_to_numpy(py, batch.token_type_ids)?)?;
        fields.set_item("position_ids", tensor_to_numpy(py, batch.position_ids)?)?;
        fields.set_item(
            "labels",
            batch
                .labels
                .map(|labels| tensor_to_numpy(py, labels))
                .transpose()?,
        )?;
        fields.set_item("lengths", batch.lengths)?;
        fields.set_item("source_indices", batch.source_indices)?;
        Ok(Some(fields))
    }
}
