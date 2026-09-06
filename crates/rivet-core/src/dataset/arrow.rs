use crate::dataset::Dataset;
use crate::errors::{RivetError, RivetResult, invalid_argument};
use crate::sample::EncodedImageSample;
use arrow::array::{Array, BinaryArray, Int32Array, Int64Array, LargeBinaryArray, StructArray};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use std::fs::File;
use std::path::PathBuf;

struct BatchMeta {
    row_start: usize,
    row_count: usize,
}

pub struct ArrowImageDatasetCore {
    batches: Vec<RecordBatch>,
    batch_meta: Vec<BatchMeta>,
    len: usize,
    image_column: String,
    label_column: String,
}

impl ArrowImageDatasetCore {
    pub fn new(
        arrow_files: Vec<PathBuf>,
        image_column: String,
        label_column: String,
    ) -> RivetResult<Self> {
        if arrow_files.is_empty() {
            return Err(invalid_argument("arrow_files must not be empty"));
        }

        let mut batches = Vec::new();
        let mut batch_meta = Vec::new();
        let mut len = 0usize;

        for path in arrow_files {
            let file = File::open(&path)?;
            let reader = StreamReader::try_new(file, None)?;

            for batch in reader {
                let batch = batch?;
                validate_image_batch(&batch, &image_column, &label_column)?;

                let row_count = batch.num_rows();
                batch_meta.push(BatchMeta {
                    row_start: len,
                    row_count,
                });
                len += row_count;
                batches.push(batch);
            }
        }

        Ok(Self {
            batches,
            batch_meta,
            len,
            image_column,
            label_column,
        })
    }

    fn locate_row(&self, index: usize) -> RivetResult<(usize, usize)> {
        if index >= self.len {
            return Err(RivetError::IndexOutOfRange {
                index,
                len: self.len,
            });
        }

        let batch_index = self
            .batch_meta
            .partition_point(|meta| meta.row_start <= index)
            .saturating_sub(1);
        let meta = &self.batch_meta[batch_index];

        if index >= meta.row_start + meta.row_count {
            return Err(RivetError::IndexOutOfRange {
                index,
                len: self.len,
            });
        }

        Ok((batch_index, index - meta.row_start))
    }
}

impl Dataset for ArrowImageDatasetCore {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> RivetResult<Self::Item> {
        let (batch_index, row) = self.locate_row(index)?;
        let batch = &self.batches[batch_index];

        Ok(EncodedImageSample {
            image: image_bytes_at(batch, &self.image_column, row)?.to_vec(),
            label: label_at(batch, &self.label_column, row)?,
        })
    }
}

fn validate_image_batch(
    batch: &RecordBatch,
    image_column: &str,
    label_column: &str,
) -> RivetResult<()> {
    let image_index = batch
        .schema()
        .index_of(image_column)
        .map_err(invalid_argument)?;
    let image = batch.column(image_index);
    let image = image
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            invalid_argument(format!(
                "{image_column} must be a struct column with a bytes field, got {:?}",
                image.data_type()
            ))
        })?;
    let bytes = image
        .column_by_name("bytes")
        .ok_or_else(|| invalid_argument(format!("{image_column}.bytes field is missing")))?;

    if !bytes.as_any().is::<BinaryArray>() && !bytes.as_any().is::<LargeBinaryArray>() {
        return Err(invalid_argument(format!(
            "{image_column}.bytes must be binary or large_binary, got {:?}",
            bytes.data_type()
        )));
    }

    let label_index = batch
        .schema()
        .index_of(label_column)
        .map_err(invalid_argument)?;
    let labels = batch.column(label_index);
    if !labels.as_any().is::<Int64Array>() && !labels.as_any().is::<Int32Array>() {
        return Err(invalid_argument(format!(
            "{label_column} must be int64 or int32, got {:?}",
            labels.data_type()
        )));
    }

    Ok(())
}

fn label_at(batch: &RecordBatch, label_column: &str, row: usize) -> RivetResult<i64> {
    let label_index = batch
        .schema()
        .index_of(label_column)
        .map_err(invalid_argument)?;
    let labels = batch.column(label_index);

    if let Some(labels) = labels.as_any().downcast_ref::<Int64Array>() {
        if labels.is_null(row) {
            return Err(invalid_argument(format!(
                "{label_column} is null at row {row}"
            )));
        }
        return Ok(labels.value(row));
    }

    if let Some(labels) = labels.as_any().downcast_ref::<Int32Array>() {
        if labels.is_null(row) {
            return Err(invalid_argument(format!(
                "{label_column} is null at row {row}"
            )));
        }
        return Ok(i64::from(labels.value(row)));
    }

    Err(invalid_argument(format!(
        "{label_column} must be int64 or int32, got {:?}",
        labels.data_type()
    )))
}

fn image_bytes_at<'a>(
    batch: &'a RecordBatch,
    image_column: &str,
    row: usize,
) -> RivetResult<&'a [u8]> {
    let image_index = batch
        .schema()
        .index_of(image_column)
        .map_err(invalid_argument)?;
    let image = batch.column(image_index);

    let image = image
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            invalid_argument(format!(
                "{image_column} must be a struct column with a bytes field, got {:?}",
                image.data_type()
            ))
        })?;

    let bytes = image
        .column_by_name("bytes")
        .ok_or_else(|| invalid_argument(format!("{image_column}.bytes field is missing")))?;

    if let Some(bytes) = bytes.as_any().downcast_ref::<BinaryArray>() {
        if bytes.is_null(row) {
            return Err(invalid_argument(format!(
                "{image_column}.bytes is null at row {row}"
            )));
        }
        return Ok(bytes.value(row));
    }

    if let Some(bytes) = bytes.as_any().downcast_ref::<LargeBinaryArray>() {
        if bytes.is_null(row) {
            return Err(invalid_argument(format!(
                "{image_column}.bytes is null at row {row}"
            )));
        }
        return Ok(bytes.value(row));
    }

    Err(invalid_argument(format!(
        "{image_column}.bytes must be binary or large_binary, got {:?}",
        bytes.data_type()
    )))
}
