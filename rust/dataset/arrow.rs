use crate::dataset::Dataset;
use crate::errors::{io_err, runtime_err, value_err};
use crate::sample::EncodedImageSample;
use arrow::array::{Array, BinaryArray, Int32Array, Int64Array, LargeBinaryArray, StructArray};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use pyo3::prelude::*;
use std::fs::File;
use std::path::PathBuf;

#[derive(Clone, Copy)]
struct RowLocation {
    batch_index: usize,
    row: usize,
}

pub(crate) struct ArrowImageDatasetCore {
    batches: Vec<RecordBatch>,
    row_locations: Vec<RowLocation>,
    image_column: String,
    label_column: String,
}

impl ArrowImageDatasetCore {
    pub(crate) fn new(
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
