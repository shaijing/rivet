use crate::dataset::Dataset;
use crate::errors::{RivetError, RivetResult, invalid_argument};
use crate::sample::EncodedImageSample;
use arrow::array::{Array, BinaryArray, Int32Array, Int64Array, LargeBinaryArray, StructArray};
use arrow::record_batch::RecordBatch;
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
use bytes::Bytes;
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

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
            for batch in decode_arrow_file(&path)? {
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

/// Decode an Arrow IPC stream file into record batches whose array buffers
/// stay backed by a read-only mmap of the file.
///
/// Ownership chain: `Mmap` -> `Bytes::from_owner` -> `Buffer`. Every record
/// batch buffer sliced out by [`StreamDecoder`] shares that `Buffer`'s
/// allocation, so the batches keep the mmap alive via refcount without the
/// dataset having to store the `Mmap` itself.
///
/// `require_alignment: true` makes the decoder fail on misaligned IPC
/// buffers instead of silently copying them to an aligned heap buffer;
/// tests use it to prove the mmap path stays zero-copy.
fn decode_arrow_file(path: &Path) -> RivetResult<Vec<RecordBatch>> {
    decode_arrow_file_with_alignment(path, false)
}

fn decode_arrow_file_with_alignment(
    path: &Path,
    require_alignment: bool,
) -> RivetResult<Vec<RecordBatch>> {
    let file = File::open(path)?;

    // SAFETY: the mapping is read-only. The mapped file must not be truncated
    // or written by others while the returned batches keep the mapping alive.
    let mmap = unsafe { Mmap::map(&file)? };

    let bytes = Bytes::from_owner(mmap);
    let mut buffer = Buffer::from(bytes);
    let mut decoder = StreamDecoder::new().with_require_alignment(require_alignment);
    let mut batches = Vec::new();

    while !buffer.is_empty() {
        if let Some(batch) = decoder.decode(&mut buffer)? {
            batches.push(batch);
        }
    }
    decoder.finish()?;

    Ok(batches)
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
            image: image_buffer_at(batch, &self.image_column, row)?,
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

/// Return row's encoded bytes as a zero-copy slice of the column's values
/// buffer: only offsets/length metadata is produced, the payload bytes are
/// shared with whatever backs the record batch (mmap, heap, network).
fn image_buffer_at(
    batch: &RecordBatch,
    image_column: &str,
    row: usize,
) -> RivetResult<Buffer> {
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
        let offsets = bytes.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        return Ok(bytes.values().slice_with_length(start, end - start));
    }

    if let Some(bytes) = bytes.as_any().downcast_ref::<LargeBinaryArray>() {
        if bytes.is_null(row) {
            return Err(invalid_argument(format!(
                "{image_column}.bytes is null at row {row}"
            )));
        }
        let offsets = bytes.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        return Ok(bytes.values().slice_with_length(start, end - start));
    }

    Err(invalid_argument(format!(
        "{image_column}.bytes must be binary or large_binary, got {:?}",
        bytes.data_type()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ArrayRef;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_ipc::writer::StreamWriter;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TMP: AtomicUsize = AtomicUsize::new(0);

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rivet-buffer-{}-{name}-{}",
            std::process::id(),
            NEXT_TMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn encoded_batch(rows: &[(Option<&[u8]>, i64)]) -> RecordBatch {
        // bytes is nullable so fixtures can carry null rows; the field
        // nullability must mirror the struct child's so StructArray::from
        // accepts unmasked child nulls.
        let bytes_field = Arc::new(Field::new("bytes", DataType::Binary, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("img", DataType::Struct(vec![bytes_field.clone()].into()), true),
            Field::new("label", DataType::Int64, false),
        ]));
        let bytes: BinaryArray = rows.iter().map(|(image, _)| *image).collect();
        let labels: Int64Array = rows.iter().map(|(_, label)| Some(*label)).collect();
        let bytes_ref: ArrayRef = Arc::new(bytes);
        let img = StructArray::from(vec![(bytes_field, bytes_ref)]);
        RecordBatch::try_new(schema, vec![Arc::new(img), Arc::new(labels)]).unwrap()
    }

    fn write_stream(dir: &Path, name: &str, batches: &[&RecordBatch]) -> PathBuf {
        let path = dir.join(name);
        let mut sink = Vec::new();
        let mut writer = StreamWriter::try_new(&mut sink, &batches[0].schema()).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
        std::fs::write(&path, sink).unwrap();
        path
    }

    fn values_bounds(batch: &RecordBatch) -> (usize, usize) {
        let image = batch
            .column_by_name("img")
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let bytes = image
            .column_by_name("bytes")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let values = bytes.values();
        let start = values.as_ptr() as usize;
        (start, start + values.len())
    }

    /// `image_buffer_at` must return a slice of the column's values buffer:
    /// same content, address inside the parent buffer, no payload copy.
    #[test]
    fn image_buffer_slices_values_without_copy() {
        let row0: &[u8] = &[1, 2, 3, 4];
        let row1: &[u8] = &[];
        let row2: &[u8] = &[5, 6];
        let batch = encoded_batch(&[(Some(row0), 10), (Some(row1), 20), (Some(row2), 30)]);
        let (values_start, values_end) = values_bounds(&batch);

        for (row, expected) in [row0, row1, row2].iter().enumerate() {
            let image = image_buffer_at(&batch, "img", row).unwrap();
            assert_eq!(image.as_slice(), *expected, "row {row} content");
            assert_eq!(label_at(&batch, "label", row).unwrap(), (row as i64 + 1) * 10);

            if !expected.is_empty() {
                let ptr = image.as_ptr() as usize;
                assert!(
                    ptr >= values_start && ptr + image.len() <= values_end,
                    "row {row} buffer must point inside the parent values buffer"
                );
            }
        }
    }

    #[test]
    fn image_buffer_at_rejects_null_bytes() {
        let row: &[u8] = &[1, 2];
        let batch = encoded_batch(&[(Some(row), 1), (None, 2)]);
        assert!(image_buffer_at(&batch, "img", 0).is_ok());
        let err = image_buffer_at(&batch, "img", 1).unwrap_err();
        assert!(err.to_string().contains("null"), "got: {err}");
    }

    /// Full chain: IPC file -> mmap -> Buffer -> StreamDecoder -> batches ->
    /// sample slices. Sample byte ranges must stay inside the mmap, and the
    /// dataset constructor must serve identical rows.
    #[test]
    fn arrow_samples_stay_mmap_backed() {
        let row0: &[u8] = &[1, 2, 3];
        let row1: &[u8] = &[];
        let row2: &[u8] = &[9; 64];
        let rows = [(Some(row0), 7i64), (Some(row1), 8), (Some(row2), 9)];

        let dir = tmp_dir("mmap");
        let batch = encoded_batch(&rows);
        let path = write_stream(&dir, "data.arrow", &[&batch]);

        // Reproduce the source-side decode while holding the mapping, so the
        // mmap bounds are known.
        let file = File::open(&path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let map_start = mmap.as_ptr() as usize;
        let map_end = map_start + mmap.len();

        let mut buffer = Buffer::from(Bytes::from_owner(mmap));
        let mut decoder = StreamDecoder::new();
        let mut decoded = Vec::new();
        while !buffer.is_empty() {
            if let Some(batch) = decoder.decode(&mut buffer).unwrap() {
                decoded.push(batch);
            }
        }
        decoder.finish().unwrap();

        for batch in &decoded {
            assert_eq!(batch.num_rows(), rows.len());
            for (row, (expected, _)) in rows.iter().enumerate() {
                let image = image_buffer_at(batch, "img", row).unwrap();
                assert_eq!(image.as_slice(), expected.unwrap(), "row {row} content");
                if image.len() > 0 {
                    let ptr = image.as_ptr() as usize;
                    assert!(
                        ptr >= map_start && ptr + image.len() <= map_end,
                        "row {row} sample bytes must stay inside the mmap"
                    );
                }
            }
        }
        drop(decoded);

        // End-to-end through the dataset constructor (its own mmap).
        let dataset = ArrowImageDatasetCore::new(
            vec![path.clone()],
            "img".to_string(),
            "label".to_string(),
        )
        .unwrap();
        assert_eq!(dataset.len(), rows.len());
        for (index, (expected, label)) in rows.iter().enumerate() {
            let sample = dataset.get(index).unwrap();
            assert_eq!(sample.image.as_slice(), expected.unwrap());
            assert_eq!(sample.label, *label);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Rows across multiple arrow shards keep one global index. Boundaries
    /// between batches inside a file and between files must resolve to the
    /// right (batch, row); out-of-range indices must be rejected.
    #[test]
    fn multi_file_dataset_indexes_across_shards() {
        let a: &[u8] = &[1];
        let b: &[u8] = &[2];
        let c: &[u8] = &[3];
        let d: &[u8] = &[4];
        let e: &[u8] = &[5];
        let expected: [&[u8]; 5] = [a, b, c, d, e];

        let dir = tmp_dir("multifile");
        // file0: two batches (2 + 1 rows); file1: one batch (2 rows).
        let batch0 = encoded_batch(&[(Some(a), 0), (Some(b), 1)]);
        let batch1 = encoded_batch(&[(Some(c), 2)]);
        let batch2 = encoded_batch(&[(Some(d), 3), (Some(e), 4)]);
        let file0 = write_stream(&dir, "shard0.arrow", &[&batch0, &batch1]);
        let file1 = write_stream(&dir, "shard1.arrow", &[&batch2]);

        let dataset = ArrowImageDatasetCore::new(
            vec![file0, file1],
            "img".to_string(),
            "label".to_string(),
        )
        .unwrap();
        assert_eq!(dataset.len(), expected.len());

        // Row-level boundaries: first row, last row of a batch, first row of
        // the next batch, first row of the second file, and the last row.
        for index in [0usize, 1, 2, 3, 4] {
            let sample = dataset.get(index).unwrap();
            assert_eq!(sample.image.as_slice(), expected[index], "row {index}");
            assert_eq!(sample.label, index as i64, "row {index} label");
        }

        // The dataset keeps its mmaps alive after the constructor returns:
        // repeat reads must still resolve after all decode locals are gone.
        let again = dataset.get(3).unwrap();
        assert_eq!(again.image.as_slice(), expected[3]);

        let err = match dataset.get(5) {
            Err(err) => err,
            Ok(_) => panic!("expected out-of-range error"),
        };
        assert!(err.to_string().contains("out of range"), "got: {err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Strict-alignment decoding must accept arrow-writer output without the
    /// heap-copy fallback: any misaligned buffer would surface as an error
    /// instead of a silent copy, so success proves the buffers are aligned
    /// and the mmap stays zero-copy.
    #[test]
    fn decoded_batches_satisfy_strict_alignment() {
        let row: &[u8] = &[0; 64];
        let dir = tmp_dir("alignment");
        let batch = encoded_batch(&[(Some(row), 1), (Some(row), 2), (Some(row), 3)]);
        let path = write_stream(&dir, "aligned.arrow", &[&batch]);

        let decoded = decode_arrow_file_with_alignment(&path, true).unwrap();
        assert_eq!(decoded.len(), 1);
        for index in 0..decoded[0].num_rows() {
            let image = image_buffer_at(&decoded[0], "img", index).unwrap();
            assert_eq!(image.as_slice(), &[0; 64]);
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}

