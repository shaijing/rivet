use crate::errors::{DataError, DataResult, invalid_argument};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use std::path::{Path, PathBuf};

use super::mmap::{ArrowOpenOptions, decode_mmap_arrow};
use super::row::ArrowRow;

/// mmap-backed Arrow IPC table. This type only knows schemas, record batches,
/// row locations, and typed row access; modality adapters live above it.
pub struct MmapArrowTable {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    offsets: Vec<usize>,
    len: usize,
}

impl MmapArrowTable {
    pub fn from_files<I, P>(paths: I) -> DataResult<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let files: Vec<PathBuf> = paths
            .into_iter()
            .map(|path| path.as_ref().to_path_buf())
            .collect();
        if files.is_empty() {
            return Err(invalid_argument("arrow_files must not be empty"));
        }

        let mut schema: Option<SchemaRef> = None;
        let mut batches = Vec::new();
        let mut offsets = vec![0usize];
        let mut len = 0usize;
        for path in &files {
            let decoded = decode_mmap_arrow(path, &ArrowOpenOptions::default())?;
            match &schema {
                None => schema = Some(decoded.schema.clone()),
                Some(expected) if expected.as_ref() != decoded.schema.as_ref() => {
                    return Err(invalid_argument(format!(
                        "Arrow schema mismatch in {}",
                        path.display()
                    )));
                }
                Some(_) => {}
            }
            for batch in decoded.batches {
                len += batch.num_rows();
                batches.push(batch);
                offsets.push(len);
            }
        }

        Ok(Self {
            schema: schema.ok_or_else(|| invalid_argument("no Arrow schema found"))?,
            batches,
            offsets,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn locate_row(&self, index: usize) -> DataResult<(&RecordBatch, usize)> {
        if index >= self.len {
            return Err(DataError::IndexOutOfRange {
                index,
                len: self.len,
            });
        }
        let batch_index = self
            .offsets
            .partition_point(|&offset| offset <= index)
            .saturating_sub(1);
        Ok((
            &self.batches[batch_index],
            index - self.offsets[batch_index],
        ))
    }

    pub fn row(&self, index: usize) -> DataResult<ArrowRow<'_>> {
        let (batch, row) = self.locate_row(index)?;
        Ok(ArrowRow::new(batch, row))
    }
}
