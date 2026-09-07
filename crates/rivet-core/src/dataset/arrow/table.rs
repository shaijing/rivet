use crate::dataset::source::Dataset;
use crate::errors::{RivetError, RivetResult, invalid_argument};
use crate::sample::image::EncodedImageSample;
use crate::sample::text::RawTextSample;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::mmap::{ArrowOpenOptions, decode_mmap_arrow};
use super::row::ArrowRow;

/// mmap-backed Arrow IPC stream. Batches are decoded once into structured
/// views whose data buffers stay file-backed (see the `mmap` module); this
/// layer handles storage, row location, and schema exposure only and knows
/// nothing about any modality.
///
/// Invariants:
/// - the table always has a schema (every input file shares it)
/// - `offsets[0] == 0`, `offsets.last() == len`
/// - batch `i` spans rows `[offsets[i], offsets[i + 1])`
pub struct MmapArrowTable {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    offsets: Vec<usize>,
    len: usize,
}

impl MmapArrowTable {
    pub fn from_files<I, P>(paths: I) -> RivetResult<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let files: Vec<_> = paths
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
            schema: schema.expect("at least one input file was checked above"),
            batches,
            offsets,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// The shared stream schema; adapters validate it once at construction
    /// instead of walking every batch. Every input file must match it.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Resolve a global row index to its (batch, local row).
    pub fn locate_row(&self, index: usize) -> RivetResult<(&RecordBatch, usize)> {
        if index >= self.len {
            return Err(RivetError::IndexOutOfRange {
                index,
                len: self.len,
            });
        }

        let batch_index = self
            .offsets
            .partition_point(|&offset| offset <= index)
            .saturating_sub(1);

        Ok((&self.batches[batch_index], index - self.offsets[batch_index]))
    }

    pub fn row(&self, index: usize) -> RivetResult<ArrowRow<'_>> {
        let (batch, row) = self.locate_row(index)?;
        Ok(ArrowRow::new(batch, row))
    }
}

/// Image-modality adapter over an [`MmapArrowTable`]: maps a
/// `struct<bytes: binary>` image column and an int label column onto
/// [`EncodedImageSample`]. The table layer itself stays modality-free.
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
    ) -> RivetResult<Self> {
        let table = Arc::new(MmapArrowTable::from_files(&arrow_files)?);
        validate_image_schema(table.schema(), &image_column, &label_column)?;

        Ok(Self {
            table,
            image_column,
            label_column,
        })
    }
}

impl Dataset for ArrowImageDataset {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn get(&self, index: usize) -> RivetResult<Self::Item> {
        let row = self.table.row(index)?;

        Ok(EncodedImageSample {
            image: row.struct_binary(&self.image_column, "bytes")?,
            label: row.i64(&self.label_column)?,
        })
    }
}

/// Text-modality adapter over an [`MmapArrowTable`]: maps a UTF-8 column
/// (and an optional int label column) onto [`RawTextSample`].
pub struct ArrowTextDataset {
    table: Arc<MmapArrowTable>,
    text_column: String,
    label_column: Option<String>,
}

impl ArrowTextDataset {
    pub fn new(
        arrow_files: Vec<PathBuf>,
        text_column: String,
        label_column: Option<String>,
    ) -> RivetResult<Self> {
        let table = Arc::new(MmapArrowTable::from_files(&arrow_files)?);
        validate_text_schema(
            table.schema(),
            &text_column,
            label_column.as_deref(),
        )?;

        Ok(Self {
            table,
            text_column,
            label_column,
        })
    }
}

impl Dataset for ArrowTextDataset {
    type Item = RawTextSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn get(&self, index: usize) -> RivetResult<Self::Item> {
        let row = self.table.row(index)?;

        Ok(RawTextSample {
            text: row.utf8(&self.text_column)?,
            label: match &self.label_column {
                Some(column) => Some(row.i64(column)?),
                None => None,
            },
        })
    }
}

fn validate_image_schema(
    schema: &Schema,
    image_column: &str,
    label_column: &str,
) -> RivetResult<()> {
    let image = schema
        .field_with_name(image_column)
        .map_err(invalid_argument)?;

    match image.data_type() {
        DataType::Struct(fields) => {
            let bytes = fields.iter().find(|field| field.name() == "bytes").ok_or_else(
                || invalid_argument(format!("{image_column}.bytes field is missing")),
            )?;
            if !matches!(bytes.data_type(), DataType::Binary | DataType::LargeBinary) {
                return Err(invalid_argument(format!(
                    "{image_column}.bytes must be binary or large_binary, got {:?}",
                    bytes.data_type()
                )));
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "{image_column} must be a struct column with a bytes field, got {other:?}"
            )));
        }
    }

    require_int_label(schema, label_column, "label")
}

fn validate_text_schema(
    schema: &Schema,
    text_column: &str,
    label_column: Option<&str>,
) -> RivetResult<()> {
    let text = schema
        .field_with_name(text_column)
        .map_err(invalid_argument)?;
    if !matches!(text.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
        return Err(invalid_argument(format!(
            "{text_column} must be utf8 or large_utf8, got {:?}",
            text.data_type()
        )));
    }

    if let Some(label_column) = label_column {
        require_int_label(schema, label_column, "label")?;
    }
    Ok(())
}

fn require_int_label(schema: &Schema, label_column: &str, role: &str) -> RivetResult<()> {
    let label = schema
        .field_with_name(label_column)
        .map_err(invalid_argument)?;
    if !matches!(label.data_type(), DataType::Int64 | DataType::Int32) {
        return Err(invalid_argument(format!(
            "{label_column} must be int64 or int32 for {role}, got {:?}",
            label.data_type()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::arrow::mmap::{ArrowOpenOptions, decode_mmap_arrow};
    use arrow::array::{ArrayRef, BinaryArray, Int64Array, StringArray, StructArray};
    use arrow_buffer::Buffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_ipc::reader::StreamDecoder;
    use arrow_ipc::writer::StreamWriter;
    use bytes::Bytes;
    use memmap2::Mmap;
    use std::fs::File;
    use std::path::Path;
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
                let image = ArrowRow::new(batch, row).struct_binary("img", "bytes").unwrap();
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
        let dataset = ArrowImageDataset::new(
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

        let dataset = ArrowImageDataset::new(
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

        let decoded = decode_mmap_arrow(&path, &ArrowOpenOptions { require_alignment: true }).unwrap();
        assert_eq!(decoded.batches.len(), 1);
        for index in 0..decoded.batches[0].num_rows() {
            let image = ArrowRow::new(&decoded.batches[0], index)
                .struct_binary("img", "bytes")
                .unwrap();
            assert_eq!(image.as_slice(), &[0; 64]);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// ArrowTextDataset serves UTF-8 rows zero-copy, with and without an
    /// optional label column, across the same mmap table used for images.
    #[test]
    fn text_dataset_reads_utf8_with_optional_labels() {
        let dir = tmp_dir("text");

        let text_schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new("label", DataType::Int64, false),
        ]));
        let text: StringArray = ["hello", "", "wörld", "🐇"].into_iter().map(Some).collect();
        let labels: Int64Array = [Some(1), Some(2), Some(3), Some(4)].into_iter().collect();
        let batch = RecordBatch::try_new(
            text_schema,
            vec![Arc::new(text), Arc::new(labels)],
        )
        .unwrap();
        let with_label = write_stream(&dir, "text.arrow", &[&batch]);

        let dataset = ArrowTextDataset::new(
            vec![with_label],
            "text".to_string(),
            Some("label".to_string()),
        )
        .unwrap();
        assert_eq!(dataset.len(), 4);
        for (index, (expected, label)) in ["hello", "", "wörld", "🐇"]
            .iter()
            .zip([1, 2, 3, 4])
            .enumerate()
        {
            let sample = dataset.get(index).unwrap();
            assert_eq!(
                std::str::from_utf8(sample.text.as_slice()).unwrap(),
                *expected,
                "row {index} text"
            );
            assert_eq!(sample.label, Some(label), "row {index} label");
        }

        let bare_schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::Utf8,
            false,
        )]));
        let bare: StringArray = ["only text"].into_iter().map(Some).collect();
        let bare_batch =
            RecordBatch::try_new(bare_schema, vec![Arc::new(bare)]).unwrap();
        let bare_file = write_stream(&dir, "bare.arrow", &[&bare_batch]);
        let bare_dataset = ArrowTextDataset::new(
            vec![bare_file],
            "text".to_string(),
            None,
        )
        .unwrap();
        let sample = bare_dataset.get(0).unwrap();
        assert_eq!(sample.text.as_slice(), b"only text");
        assert_eq!(sample.label, None);

        // Schema validation is constructor-time and per-table.
        assert!(ArrowTextDataset::new(
            vec![],
            "text".to_string(),
            None,
        )
        .is_err());
        assert!(ArrowTextDataset::new(
            vec![write_stream(&dir, "bare2.arrow", &[&bare_batch])],
            "missing".to_string(),
            None,
        )
        .is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    fn write_empty_stream(dir: &Path, name: &str, schema: &SchemaRef) -> PathBuf {
        let path = dir.join(name);
        let mut sink = Vec::new();
        let mut writer = StreamWriter::try_new(&mut sink, schema).unwrap();
        writer.finish().unwrap();
        std::fs::write(&path, sink).unwrap();
        path
    }

    /// IPC streams with conflicting schemas must fail at construction, not
    /// when rows from the second file are first accessed.
    #[test]
    fn mismatched_schemas_rejected_across_files() {
        let dir = tmp_dir("mismatch");

        let utf8_schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::Utf8,
            false,
        )]));
        let utf8: StringArray = ["a"].into_iter().map(Some).collect();
        let utf8_batch =
            RecordBatch::try_new(utf8_schema, vec![Arc::new(utf8)]).unwrap();

        let binary_schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::Binary,
            false,
        )]));
        let binary: BinaryArray = [Some(&b"b"[..])].into_iter().collect();
        let binary_batch =
            RecordBatch::try_new(binary_schema, vec![Arc::new(binary)]).unwrap();

        let file_a = write_stream(&dir, "utf8.arrow", &[&utf8_batch]);
        let file_b = write_stream(&dir, "binary.arrow", &[&binary_batch]);

        let err = match MmapArrowTable::from_files(vec![file_a, file_b]) {
            Err(err) => err,
            Ok(_) => panic!("expected schema mismatch error"),
        };
        assert!(err.to_string().contains("schema mismatch"), "got: {err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stream with a schema but zero batches is still a valid typed
    /// dataset (len 0, schema retained); empty and non-empty files mix.
    #[test]
    fn empty_streams_keep_schema_and_mix_with_data() {
        let dir = tmp_dir("empty");
        let text_schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::Utf8,
            false,
        )]));
        let empty = write_empty_stream(&dir, "empty.arrow", &text_schema);

        let table = MmapArrowTable::from_files(vec![empty.clone()]).unwrap();
        assert_eq!(table.len(), 0);
        assert!(table.schema().field_with_name("text").is_ok());

        let dataset =
            ArrowTextDataset::new(vec![empty.clone()], "text".to_string(), None).unwrap();
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());

        let bare: StringArray = ["only text"].into_iter().map(Some).collect();
        let bare_batch =
            RecordBatch::try_new(text_schema.clone(), vec![Arc::new(bare)]).unwrap();
        let data = write_stream(&dir, "data.arrow", &[&bare_batch]);
        let empty2 = write_empty_stream(&dir, "empty2.arrow", &text_schema);

        let mixed = ArrowTextDataset::new(
            vec![empty, data, empty2],
            "text".to_string(),
            None,
        )
        .unwrap();
        assert_eq!(mixed.len(), 1);
        let sample = mixed.get(0).unwrap();
        assert_eq!(sample.text.as_slice(), b"only text");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// offsets spans batches: starts at 0, ends at len, one entry per batch.
    #[test]
    fn offsets_track_batch_boundaries() {
        let dir = tmp_dir("offsets");
        let a: &[u8] = &[1];
        let b: &[u8] = &[2];
        let c: &[u8] = &[3];
        let batch_a = encoded_batch(&[(Some(a), 0), (Some(b), 1)]);
        let batch_b = encoded_batch(&[(Some(c), 2)]);
        let file_a = write_stream(&dir, "a.arrow", &[&batch_a]);
        let file_b = write_stream(&dir, "b.arrow", &[&batch_b]);

        let table = MmapArrowTable::from_files(vec![file_a, file_b]).unwrap();
        assert_eq!(table.offsets, vec![0, 2, 3]);
        assert_eq!(table.offsets.len(), table.batches.len() + 1);
        assert_eq!(*table.offsets.last().unwrap(), table.len);

        std::fs::remove_dir_all(&dir).ok();
    }
}
