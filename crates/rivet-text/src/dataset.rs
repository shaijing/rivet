//! Arrow text sources. Batches retain their Arrow buffers; requested rows become
//! owned strings. IPC stream loading eagerly retains all batches in memory.
use crate::TextSample;
use crate::errors::{TextResult, invalid};
use arrow::array::{Array, Int64Array, LargeStringArray, StringArray};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use rivet_data::dataset::arrow::MmapArrowTable;
use rivet_data::dataset::source::{Dataset, SourceCapabilities, validate_indices};
use rivet_data::errors::{DataResult, invalid_argument};
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextColumns {
    pub text: String,
    pub text_pair: Option<String>,
    pub label: Option<String>,
}
impl Default for TextColumns {
    fn default() -> Self {
        Self {
            text: "text".into(),
            text_pair: None,
            label: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ArrowTextDataset {
    batches: Vec<RecordBatch>,
    ends: Vec<usize>,
    columns: TextColumns,
    len: usize,
}
impl ArrowTextDataset {
    pub fn from_batches(batches: Vec<RecordBatch>, columns: TextColumns) -> TextResult<Self> {
        if columns.text.is_empty()
            || columns.text_pair.as_ref().is_some_and(String::is_empty)
            || columns.label.as_ref().is_some_and(String::is_empty)
        {
            return Err(invalid("Arrow column names cannot be empty"));
        }
        let mut ends = Vec::with_capacity(batches.len());
        let mut len = 0usize;
        for batch in &batches {
            validate_schema(&batch.schema(), &columns)?;
            len = len
                .checked_add(batch.num_rows())
                .ok_or_else(|| invalid("Arrow text row count overflow"))?;
            ends.push(len);
        }
        Ok(Self {
            batches,
            ends,
            columns,
            len,
        })
    }
    pub fn from_ipc_stream(reader: impl Read, columns: TextColumns) -> TextResult<Self> {
        let reader = StreamReader::try_new(reader, None)?;
        validate_schema(&reader.schema(), &columns)?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        Self::from_batches(batches, columns)
    }
    pub fn open_ipc(path: impl AsRef<Path>, columns: TextColumns) -> TextResult<Self> {
        let table = MmapArrowTable::from_files([path])?;
        validate_schema(table.schema(), &columns)?;
        Self::from_batches(table.batches().to_vec(), columns)
    }
    pub fn open_ipc_stream(path: impl AsRef<Path>, columns: TextColumns) -> TextResult<Self> {
        Self::from_ipc_stream(std::fs::File::open(path)?, columns)
    }
}
fn validate_schema(schema: &Schema, columns: &TextColumns) -> TextResult<()> {
    for name in std::iter::once(&columns.text).chain(columns.text_pair.iter()) {
        let column = schema
            .column_with_name(name)
            .map(|(_, field)| field)
            .ok_or_else(|| invalid(format!("missing text column {name:?}")))?;
        if !matches!(column.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
            return Err(invalid(format!(
                "text column {name:?} must be Utf8 or LargeUtf8"
            )));
        }
    }
    if let Some(name) = &columns.label {
        let column = schema
            .column_with_name(name)
            .map(|(_, field)| field)
            .ok_or_else(|| invalid(format!("missing label column {name:?}")))?;
        if column.data_type() != &DataType::Int64 {
            return Err(invalid(format!("label column {name:?} must be Int64")));
        }
    }
    Ok(())
}

fn text_at(batch: &RecordBatch, name: &str, row: usize, source_index: usize) -> DataResult<String> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| invalid_argument(format!("missing column {name:?}")))?;
    if column.is_null(row) {
        return Err(invalid_argument(format!(
            "null in text column {name:?} at source row {source_index}"
        )));
    }
    match column.data_type() {
        DataType::Utf8 => Ok(column
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("validated utf8")
            .value(row)
            .to_owned()),
        DataType::LargeUtf8 => Ok(column
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("validated large utf8")
            .value(row)
            .to_owned()),
        _ => Err(invalid_argument("invalid text column type")),
    }
}
impl Dataset for ArrowTextDataset {
    type Item = TextSample;
    fn len(&self) -> usize {
        self.len
    }
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            batched_reads: true,
            parallel_reads: true,
            ..Default::default()
        }
    }
    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<TextSample>> {
        validate_indices(indices, self.len)?;
        indices
            .iter()
            .map(|&index| {
                let batch_index = self.ends.partition_point(|&end| end <= index);
                let start = if batch_index == 0 {
                    0
                } else {
                    self.ends[batch_index - 1]
                };
                let batch = &self.batches[batch_index];
                let row = index - start;
                let text = text_at(batch, &self.columns.text, row, index)?;
                let text_pair = self
                    .columns
                    .text_pair
                    .as_ref()
                    .map(|name| text_at(batch, name, row, index))
                    .transpose()?;
                let label = self
                    .columns
                    .label
                    .as_ref()
                    .map(|name| {
                        let column = batch.column_by_name(name).expect("validated label column");
                        if column.is_null(row) {
                            return Err(invalid_argument(format!(
                                "null in label column {name:?} at source row {index}"
                            )));
                        }
                        Ok(column
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .expect("validated int64")
                            .value(row))
                    })
                    .transpose()?;
                Ok(TextSample {
                    text,
                    text_pair,
                    label,
                })
            })
            .collect()
    }
}
