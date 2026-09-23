use crate::sample::image::EncodedImageSample;
use arrow::array::{Array, BinaryArray, Int32Array, Int64Array, LargeBinaryArray, StructArray};
use arrow::datatypes::{DataType, Schema};
use arrow_buffer::Buffer;
use rivet_data::dataset::Dataset;
use rivet_data::dataset::arrow::MmapArrowTable;
use rivet_data::dataset::source::{AccessPattern, SourceCapabilities, validate_indices};
use rivet_data::errors::{DataResult, invalid_argument};
use std::path::PathBuf;
use std::sync::Arc;

/// Image adapter over the modality-neutral Arrow table.
pub struct ArrowImageDataset {
    table: Arc<MmapArrowTable>,
    image_column: String,
    label_column: String,
}

impl ArrowImageDataset {
    pub fn new(
        arrow_files: Vec<PathBuf>,
        image_column: String,
        label_column: String,
    ) -> DataResult<Self> {
        let table = Arc::new(MmapArrowTable::from_files(&arrow_files)?);
        validate_image_schema(table.schema(), &image_column, &label_column)?;
        Ok(Self {
            table,
            image_column,
            label_column,
        })
    }

    fn get_one(&self, index: usize) -> DataResult<EncodedImageSample> {
        let row = self.table.row(index)?;
        let image = match self
            .table
            .schema()
            .field_with_name(&self.image_column)
            .map_err(invalid_argument)?
            .data_type()
        {
            DataType::Binary | DataType::LargeBinary => row.binary(&self.image_column)?,
            DataType::Struct(_) => row.struct_binary(&self.image_column, "bytes")?,
            data_type => {
                return Err(invalid_argument(format!(
                    "{} must be binary, large_binary, or a struct with a bytes field, got {:?}",
                    self.image_column, data_type
                )));
            }
        };
        Ok(EncodedImageSample {
            image,
            label: row.i64(&self.label_column)?,
        })
    }

    fn get_many_individually(&self, indices: &[usize]) -> DataResult<Vec<EncodedImageSample>> {
        indices.iter().map(|&index| self.get_one(index)).collect()
    }

    fn get_many_by_batch(&self, indices: &[usize]) -> DataResult<Vec<EncodedImageSample>> {
        let groups = self.table.group_rows(indices)?;
        let mut output = std::iter::repeat_with(|| None)
            .take(indices.len())
            .collect::<Vec<Option<EncodedImageSample>>>();
        let image_type = self
            .table
            .schema()
            .field_with_name(&self.image_column)
            .map_err(invalid_argument)?
            .data_type();

        for group in groups {
            let image = image_bytes_column(group.batch, &self.image_column, image_type)?;
            let label = label_column(group.batch.column_by_name(&self.label_column).ok_or_else(
                || invalid_argument(format!("missing column {}", self.label_column)),
            )?)?;

            for requested in group.rows {
                output[requested.request_index] = Some(EncodedImageSample {
                    image: image.value(requested.batch_row, &self.image_column)?,
                    label: label.value(requested.batch_row, &self.label_column)?,
                });
            }
        }

        Ok(output
            .into_iter()
            .map(|sample| sample.expect("every requested row was assigned"))
            .collect())
    }
}

impl Dataset for ArrowImageDataset {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            access_pattern: AccessPattern::RandomAccess,
            batched_reads: true,
            preferred_batch_size: None,
            zero_copy: true,
            parallel_reads: true,
            async_reads: false,
            read_device: Some("cpu"),
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        validate_indices(indices, self.len())?;
        // Small requests avoid group allocation; larger requests amortize
        // column downcasts by processing all selected rows in one batch at a
        // time. Both paths preserve caller order and duplicate indices.
        if indices.len() < 256 {
            self.get_many_individually(indices)
        } else {
            self.get_many_by_batch(indices)
        }
    }
}

enum BinaryValues<'a> {
    Binary(&'a BinaryArray),
    LargeBinary(&'a LargeBinaryArray),
}

impl BinaryValues<'_> {
    fn value(&self, row: usize, column: &str) -> DataResult<Buffer> {
        match self {
            Self::Binary(values) => {
                if values.is_null(row) {
                    return Err(invalid_argument(format!("{column} is null at row {row}")));
                }
                Ok(slice_binary(values.values(), values.value_offsets(), row))
            }
            Self::LargeBinary(values) => {
                if values.is_null(row) {
                    return Err(invalid_argument(format!("{column} is null at row {row}")));
                }
                Ok(slice_binary(values.values(), values.value_offsets(), row))
            }
        }
    }
}

enum LabelValues<'a> {
    Int64(&'a Int64Array),
    Int32(&'a Int32Array),
}

impl LabelValues<'_> {
    fn value(&self, row: usize, column: &str) -> DataResult<i64> {
        match self {
            Self::Int64(values) if !values.is_null(row) => Ok(values.value(row)),
            Self::Int32(values) if !values.is_null(row) => Ok(i64::from(values.value(row))),
            _ => Err(invalid_argument(format!("{column} is null at row {row}"))),
        }
    }
}

fn image_bytes_column<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    column: &str,
    data_type: &DataType,
) -> DataResult<BinaryValues<'a>> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| invalid_argument(format!("missing column {column}")))?;
    if let DataType::Struct(_) = data_type {
        let image = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| invalid_argument(format!("{column} must be a struct column")))?;
        let bytes = image
            .column_by_name("bytes")
            .ok_or_else(|| invalid_argument(format!("{column}.bytes field is missing")))?;
        binary_values(bytes.as_ref(), &format!("{column}.bytes"))
    } else {
        binary_values(array.as_ref(), column)
    }
}

fn binary_values<'a>(array: &'a dyn Array, column: &str) -> DataResult<BinaryValues<'a>> {
    if let Some(values) = array.as_any().downcast_ref::<BinaryArray>() {
        return Ok(BinaryValues::Binary(values));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(BinaryValues::LargeBinary(values));
    }
    Err(invalid_argument(format!(
        "{column} must be binary or large_binary, got {:?}",
        array.data_type()
    )))
}

fn label_column(array: &dyn Array) -> DataResult<LabelValues<'_>> {
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(LabelValues::Int64(values));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(LabelValues::Int32(values));
    }
    Err(invalid_argument(format!(
        "label must be int64 or int32, got {:?}",
        array.data_type()
    )))
}

trait Offset: Copy {
    fn as_usize(self) -> usize;
}

impl Offset for i32 {
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl Offset for i64 {
    fn as_usize(self) -> usize {
        self as usize
    }
}

fn slice_binary<O: Offset>(values: &Buffer, offsets: &[O], row: usize) -> Buffer {
    let start = offsets[row].as_usize();
    let end = offsets[row + 1].as_usize();
    values.slice_with_length(start, end - start)
}

fn validate_image_schema(
    schema: &Schema,
    image_column: &str,
    label_column: &str,
) -> DataResult<()> {
    let image = schema
        .field_with_name(image_column)
        .map_err(invalid_argument)?;
    match image.data_type() {
        DataType::Binary | DataType::LargeBinary => {}
        DataType::Struct(fields) => {
            let bytes = fields
                .iter()
                .find(|field| field.name() == "bytes")
                .ok_or_else(|| {
                    invalid_argument(format!("{image_column}.bytes field is missing"))
                })?;
            if !matches!(bytes.data_type(), DataType::Binary | DataType::LargeBinary) {
                return Err(invalid_argument(format!(
                    "{image_column}.bytes must be binary or large_binary, got {:?}",
                    bytes.data_type()
                )));
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "{image_column} must be binary, large_binary, or a struct with a bytes field, got {other:?}"
            )));
        }
    }

    let label = schema
        .field_with_name(label_column)
        .map_err(invalid_argument)?;
    if !matches!(label.data_type(), DataType::Int64 | DataType::Int32) {
        return Err(invalid_argument(format!(
            "{label_column} must be int64 or int32, got {:?}",
            label.data_type()
        )));
    }
    Ok(())
}
