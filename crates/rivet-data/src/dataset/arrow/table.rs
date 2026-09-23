use crate::errors::{DataError, DataResult, invalid_argument};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use std::path::{Path, PathBuf};

use super::mmap::{ArrowOpenOptions, decode_mmap_arrow};
use super::row::ArrowRow;

/// One requested row's position within an [`ArrowRowGroup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestedRow {
    /// Position in the original input index slice.
    pub request_index: usize,
    /// Position within the group's record batch.
    pub batch_row: usize,
}

/// Input rows that belong to the same Arrow record batch.
pub struct ArrowRowGroup<'a> {
    /// The shared record batch containing the requested rows.
    pub batch: &'a RecordBatch,
    /// Requested positions within `batch`; duplicates are retained.
    pub rows: Vec<RequestedRow>,
}

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
        let batch_index = self.batch_index(index);
        Ok((
            &self.batches[batch_index],
            index - self.offsets[batch_index],
        ))
    }

    pub fn row(&self, index: usize) -> DataResult<ArrowRow<'_>> {
        let (batch, row) = self.locate_row(index)?;
        Ok(ArrowRow::new(batch, row))
    }

    /// Resolve multiple logical rows together, grouping requests by record
    /// batch while preserving the input order and duplicate indices.
    ///
    /// This avoids repeated public row lookups for callers processing a
    /// collection of indices. Each returned [`ArrowRow`] remains a zero-copy
    /// view into its mmap-backed record batch.
    pub fn rows(&self, indices: &[usize]) -> DataResult<Vec<ArrowRow<'_>>> {
        let groups = self.group_rows(indices)?;
        let mut output = std::iter::repeat_with(|| None)
            .take(indices.len())
            .collect::<Vec<Option<ArrowRow<'_>>>>();
        for group in groups {
            for requested in group.rows {
                output[requested.request_index] =
                    Some(ArrowRow::new(group.batch, requested.batch_row));
            }
        }

        Ok(output
            .into_iter()
            .map(|row| row.expect("every requested row was assigned"))
            .collect())
    }

    /// Map requested indices to record batches, retaining each original
    /// request position so consumers can process columns batch-by-batch and
    /// reconstruct results in request order.
    pub fn group_rows(&self, indices: &[usize]) -> DataResult<Vec<ArrowRowGroup<'_>>> {
        let mut grouped = (0..self.batches.len())
            .map(|_| Vec::<(usize, usize)>::new())
            .collect::<Vec<_>>();

        for (output_index, &index) in indices.iter().enumerate() {
            if index >= self.len {
                return Err(DataError::IndexOutOfRange {
                    index,
                    len: self.len,
                });
            }
            let batch_index = self.batch_index(index);
            grouped[batch_index].push((output_index, index - self.offsets[batch_index]));
        }

        Ok(grouped
            .into_iter()
            .enumerate()
            .filter(|(_, requests)| !requests.is_empty())
            .map(|(batch_index, requests)| ArrowRowGroup {
                batch: &self.batches[batch_index],
                rows: requests
                    .into_iter()
                    .map(|(request_index, batch_row)| RequestedRow {
                        request_index,
                        batch_row,
                    })
                    .collect(),
            })
            .collect())
    }

    fn batch_index(&self, index: usize) -> usize {
        self.offsets
            .partition_point(|&offset| offset <= index)
            .saturating_sub(1)
    }
}
