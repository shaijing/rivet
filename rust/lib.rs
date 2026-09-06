use arrow::array::{Array, BinaryArray, Int32Array, Int64Array, LargeBinaryArray, StructArray};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> PyResult<Self::Item>;
}

#[derive(Clone, Copy)]
struct RowLocation {
    batch_index: usize,
    row: usize,
}

struct EncodedImageSample {
    image: Vec<u8>,
    label: i64,
}

struct DecodedSample {
    image: Vec<u8>,
    width: u32,
    height: u32,
    channels: u8,
    label: i64,
}

struct ArrowImageDatasetCore {
    batches: Vec<RecordBatch>,
    row_locations: Vec<RowLocation>,
    image_column: String,
    label_column: String,
}

impl ArrowImageDatasetCore {
    fn new(
        arrow_files: Vec<PathBuf>,
        image_column: String,
        label_column: String,
    ) -> PyResult<Self> {
        if arrow_files.is_empty() {
            return Err(value_err("arrow_files must not be empty"));
        }

        let mut batches = Vec::new();
        let mut row_locations = Vec::new();

        for path in arrow_files {
            let file = File::open(&path).map_err(io_err)?;
            let reader = StreamReader::try_new(file, None).map_err(runtime_err)?;

            for batch in reader {
                let batch = batch.map_err(runtime_err)?;
                validate_image_batch(&batch, &image_column, &label_column)?;

                let batch_index = batches.len();
                row_locations
                    .extend((0..batch.num_rows()).map(|row| RowLocation { batch_index, row }));
                batches.push(batch);
            }
        }

        Ok(Self {
            batches,
            row_locations,
            image_column,
            label_column,
        })
    }
}

impl Dataset for ArrowImageDatasetCore {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.row_locations.len()
    }

    fn get(&self, index: usize) -> PyResult<Self::Item> {
        let location = self
            .row_locations
            .get(index)
            .ok_or_else(|| value_err(format!("index {index} is out of range")))?;
        let batch = &self.batches[location.batch_index];

        Ok(EncodedImageSample {
            image: image_bytes_at(batch, &self.image_column, location.row)?.to_vec(),
            label: label_at(batch, &self.label_column, location.row)?,
        })
    }
}

struct ImageBatch {
    images: Vec<u8>,
    labels: Vec<i64>,
    shape: (usize, usize, usize, usize),
}

#[derive(Clone)]
enum PipelineOp {
    DecodeImage,
    Batch { size: usize },
}

#[derive(Clone)]
struct ImagePipeline {
    dataset: Arc<ArrowImageDatasetCore>,
    ops: Vec<PipelineOp>,
}

impl ImagePipeline {
    fn new(dataset: Arc<ArrowImageDatasetCore>) -> Self {
        Self {
            dataset,
            ops: Vec::new(),
        }
    }

    fn decode_image(mut self) -> Self {
        self.ops.push(PipelineOp::DecodeImage);
        self
    }

    fn batch(mut self, size: usize) -> PyResult<Self> {
        if size == 0 {
            return Err(value_err("batch size must be greater than 0"));
        }

        self.ops.push(PipelineOp::Batch { size });
        Ok(self)
    }

    fn compile(self) -> PyResult<ImageDataLoader> {
        let mut batch_size = None;

        for op in &self.ops {
            match op {
                PipelineOp::DecodeImage => {}
                PipelineOp::Batch { size } => batch_size = Some(*size),
            }
        }

        let batch_size = batch_size.ok_or_else(|| value_err("pipeline requires .batch(size)"))?;
        let len = self.dataset.len();

        Ok(ImageDataLoader {
            pipeline: Arc::new(self),
            sampler: SequentialSampler::new(len),
            batch_size,
        })
    }
}

struct ImageBatchBuilder {
    images: Vec<u8>,
    labels: Vec<i64>,
    expected_shape: Option<(u32, u32, u8)>,
}

impl ImageBatchBuilder {
    fn with_capacity(batch_size: usize) -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::with_capacity(batch_size),
            expected_shape: None,
        }
    }

    fn push(&mut self, sample: DecodedSample) -> PyResult<()> {
        let shape = (sample.height, sample.width, sample.channels);

        match self.expected_shape {
            Some(expected) if expected != shape => Err(value_err(format!(
                "all images in a batch must have the same shape; expected {:?}, got {:?}",
                expected, shape
            ))),
            Some(_) => {
                self.images.extend_from_slice(&sample.image);
                self.labels.push(sample.label);
                Ok(())
            }
            None => {
                self.expected_shape = Some(shape);
                self.images.extend_from_slice(&sample.image);
                self.labels.push(sample.label);
                Ok(())
            }
        }
    }

    fn finish(self) -> ImageBatch {
        let batch_len = self.labels.len();
        let (height, width, channels) = self.expected_shape.unwrap_or((0, 0, 3));

        ImageBatch {
            images: self.images,
            labels: self.labels,
            shape: (
                batch_len,
                height as usize,
                width as usize,
                channels as usize,
            ),
        }
    }
}

struct SequentialSampler {
    position: usize,
    len: usize,
}

impl SequentialSampler {
    fn new(len: usize) -> Self {
        Self { position: 0, len }
    }

    fn next_indices(&mut self, batch_size: usize) -> Option<Vec<usize>> {
        if self.position >= self.len {
            return None;
        }

        let end = (self.position + batch_size).min(self.len);
        let indices = (self.position..end).collect();
        self.position = end;
        Some(indices)
    }
}

struct ImageDataLoader {
    pipeline: Arc<ImagePipeline>,
    batch_size: usize,
    sampler: SequentialSampler,
}

impl ImageDataLoader {
    fn new(dataset: Arc<ArrowImageDatasetCore>, batch_size: usize) -> PyResult<Self> {
        ImagePipeline::new(dataset)
            .decode_image()
            .batch(batch_size)?
            .compile()
    }

    fn next_batch(&mut self) -> PyResult<Option<ImageBatch>> {
        let Some(indices) = self.sampler.next_indices(self.batch_size) else {
            return Ok(None);
        };

        let mut batch = ImageBatchBuilder::with_capacity(indices.len());

        for index in indices {
            let encoded = self.pipeline.dataset.get(index)?;
            let decoded = self.decode_sample(encoded)?;
            batch.push(decoded)?;
        }

        Ok(Some(batch.finish()))
    }

    fn decode_sample(&self, encoded: EncodedImageSample) -> PyResult<DecodedSample> {
        let mut decoded = None;

        for op in &self.pipeline.ops {
            match op {
                PipelineOp::DecodeImage => {
                    decoded = Some(decode_rgb(&encoded.image, encoded.label)?);
                }
                PipelineOp::Batch { .. } => {}
            }
        }

        decoded.ok_or_else(|| value_err("image must be decoded before batching"))
    }
}

fn io_err(err: std::io::Error) -> PyErr {
    PyIOError::new_err(err.to_string())
}

fn runtime_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn value_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn decode_rgb(encoded: &[u8], label: i64) -> PyResult<DecodedSample> {
    let img = image::load_from_memory(encoded)
        .map_err(runtime_err)?
        .to_rgb8();
    let (width, height) = img.dimensions();

    Ok(DecodedSample {
        image: img.into_raw(),
        width,
        height,
        channels: 3,
        label,
    })
}

fn validate_image_batch(
    batch: &RecordBatch,
    image_column: &str,
    label_column: &str,
) -> PyResult<()> {
    let image_index = batch.schema().index_of(image_column).map_err(value_err)?;
    let image = batch.column(image_index);
    let image = image
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            value_err(format!(
                "{image_column} must be a struct column with a bytes field, got {:?}",
                image.data_type()
            ))
        })?;
    let bytes = image
        .column_by_name("bytes")
        .ok_or_else(|| value_err(format!("{image_column}.bytes field is missing")))?;

    if !bytes.as_any().is::<BinaryArray>() && !bytes.as_any().is::<LargeBinaryArray>() {
        return Err(value_err(format!(
            "{image_column}.bytes must be binary or large_binary, got {:?}",
            bytes.data_type()
        )));
    }

    let label_index = batch.schema().index_of(label_column).map_err(value_err)?;
    let labels = batch.column(label_index);
    if !labels.as_any().is::<Int64Array>() && !labels.as_any().is::<Int32Array>() {
        return Err(value_err(format!(
            "{label_column} must be int64 or int32, got {:?}",
            labels.data_type()
        )));
    }

    Ok(())
}

fn label_at(batch: &RecordBatch, label_column: &str, row: usize) -> PyResult<i64> {
    let label_index = batch.schema().index_of(label_column).map_err(value_err)?;
    let labels = batch.column(label_index);

    if let Some(labels) = labels.as_any().downcast_ref::<Int64Array>() {
        if labels.is_null(row) {
            return Err(value_err(format!("{label_column} is null at row {row}")));
        }
        return Ok(labels.value(row));
    }

    if let Some(labels) = labels.as_any().downcast_ref::<Int32Array>() {
        if labels.is_null(row) {
            return Err(value_err(format!("{label_column} is null at row {row}")));
        }
        return Ok(i64::from(labels.value(row)));
    }

    Err(value_err(format!(
        "{label_column} must be int64 or int32, got {:?}",
        labels.data_type()
    )))
}

fn image_bytes_at<'a>(
    batch: &'a RecordBatch,
    image_column: &str,
    row: usize,
) -> PyResult<&'a [u8]> {
    let image_index = batch.schema().index_of(image_column).map_err(value_err)?;
    let image = batch.column(image_index);

    let image = image
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            value_err(format!(
                "{image_column} must be a struct column with a bytes field, got {:?}",
                image.data_type()
            ))
        })?;

    let bytes = image
        .column_by_name("bytes")
        .ok_or_else(|| value_err(format!("{image_column}.bytes field is missing")))?;

    if let Some(bytes) = bytes.as_any().downcast_ref::<BinaryArray>() {
        if bytes.is_null(row) {
            return Err(value_err(format!(
                "{image_column}.bytes is null at row {row}"
            )));
        }
        return Ok(bytes.value(row));
    }

    if let Some(bytes) = bytes.as_any().downcast_ref::<LargeBinaryArray>() {
        if bytes.is_null(row) {
            return Err(value_err(format!(
                "{image_column}.bytes is null at row {row}"
            )));
        }
        return Ok(bytes.value(row));
    }

    Err(value_err(format!(
        "{image_column}.bytes must be binary or large_binary, got {:?}",
        bytes.data_type()
    )))
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

#[pyclass(name = "_ArrowDataset")]
struct PyArrowDataset {
    inner: Arc<ArrowImageDatasetCore>,
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
struct PyImagePipeline {
    inner: ImagePipeline,
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
struct PyDataLoader {
    inner: ImageDataLoader,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    fn new(dataset: &PyArrowDataset, batch_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: ImageDataLoader::new(Arc::clone(&dataset.inner), batch_size)?,
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
fn read_image_batch(
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
    if batch_size == 0 {
        return Err(value_err("batch_size must be greater than 0"));
    }

    let dataset = Arc::new(ArrowImageDatasetCore::new(
        arrow_files,
        image_column.to_string(),
        label_column.to_string(),
    )?);
    let mut loader = ImageDataLoader {
        batch_size,
        sampler: SequentialSampler {
            position: start.min(dataset.len()),
            len: dataset.len(),
        },
        pipeline: Arc::new(
            ImagePipeline::new(dataset)
                .decode_image()
                .batch(batch_size)?,
        ),
    };
    let batch = loader
        .next_batch()?
        .unwrap_or_else(|| ImageBatchBuilder::with_capacity(0).finish());

    image_batch_to_py(py, batch)
}

#[pymodule]
fn _rivet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArrowDataset>()?;
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyImagePipeline>()?;
    m.add_function(wrap_pyfunction!(read_image_batch, m)?)?;
    Ok(())
}
