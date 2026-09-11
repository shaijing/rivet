use super::LanceTable;
use crate::dataset::arrow::ArrowRow;
use crate::dataset::source::{Dataset, validate_indices};
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::EncodedImageSample;
use arrow::datatypes::{DataType, Schema};
use std::path::Path;
use std::sync::Arc;

/// Rivet's native Lance image classification dataset.
///
/// The hot-path schema is intentionally small:
/// `image: binary | large_binary` and `label: int32 | int64`.
/// Additional Lance columns are ignored through projection.
pub struct LanceImageDataset {
    table: Arc<LanceTable>,
    image_column: String,
    label_column: String,
}

impl LanceImageDataset {
    pub fn open(
        path: impl AsRef<Path>,
        image_column: impl Into<String>,
        label_column: impl Into<String>,
    ) -> RivetResult<Self> {
        let table = Arc::new(LanceTable::open(path)?);
        let image_column = image_column.into();
        let label_column = label_column.into();
        validate_schema(table.schema(), &image_column, &label_column)?;

        Ok(Self {
            table,
            image_column,
            label_column,
        })
    }

    pub fn new(
        path: impl AsRef<Path>,
        image_column: impl Into<String>,
        label_column: impl Into<String>,
    ) -> RivetResult<Self> {
        Self::open(path, image_column, label_column)
    }

    pub fn open_default(path: impl AsRef<Path>) -> RivetResult<Self> {
        Self::open(path, "image", "label")
    }

    pub fn image_column(&self) -> &str {
        &self.image_column
    }

    pub fn label_column(&self) -> &str {
        &self.label_column
    }

    fn get_one(
        &self,
        batch: &arrow_array::RecordBatch,
        row: usize,
    ) -> RivetResult<EncodedImageSample> {
        let row = ArrowRow::new(batch, row);
        Ok(EncodedImageSample {
            // ArrowRow slices the BinaryArray values buffer, so this keeps
            // Lance's encoded payload allocation alive without copying bytes.
            image: row.binary(&self.image_column)?,
            label: row.i64(&self.label_column)?,
        })
    }
}

impl Dataset for LanceImageDataset {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        validate_indices(indices, self.len())?;

        let batch = self
            .table
            .take(indices, &[&self.image_column, &self.label_column])?;
        if batch.num_rows() != indices.len() {
            return Err(invalid_argument(format!(
                "Lance take returned {} rows for {} indices",
                batch.num_rows(),
                indices.len()
            )));
        }

        (0..batch.num_rows())
            .map(|row| self.get_one(&batch, row))
            .collect()
    }
}

fn validate_schema(schema: &Schema, image_column: &str, label_column: &str) -> RivetResult<()> {
    let image = schema
        .field_with_name(image_column)
        .map_err(invalid_argument)?;
    if !matches!(image.data_type(), DataType::Binary | DataType::LargeBinary) {
        return Err(invalid_argument(format!(
            "{image_column} must be binary or large_binary, got {:?}",
            image.data_type()
        )));
    }

    let label = schema
        .field_with_name(label_column)
        .map_err(invalid_argument)?;
    if !matches!(label.data_type(), DataType::Int32 | DataType::Int64) {
        return Err(invalid_argument(format!(
            "{label_column} must be int32 or int64, got {:?}",
            label.data_type()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatchIterator;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_lance_path() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rivet-lance-test-{}-{}",
            std::process::id(),
            NEXT_TMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_fixture(path: &std::path::Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("image", DataType::Binary, false),
            Field::new("label", DataType::Int32, false),
            Field::new("ignored", DataType::Utf8, false),
        ]));
        let image: BinaryArray = [Some(&b"zero"[..]), Some(&b"one"[..]), Some(&b"two"[..])]
            .into_iter()
            .collect();
        let label = Int32Array::from(vec![0, 1, 2]);
        let ignored = arrow::array::StringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(image), Arc::new(label), Arc::new(ignored)],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(lance::Dataset::write(reader, path.to_str().unwrap(), None))
            .unwrap();
    }

    #[test]
    fn reads_ordered_duplicate_rows_with_one_batch_request() {
        let path = temp_lance_path();
        write_fixture(&path);

        let dataset = LanceImageDataset::open_default(&path).unwrap();
        assert_eq!(dataset.len(), 3);
        let samples = dataset.get_many(&[2, 0, 2]).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].image.as_slice(), b"two");
        assert_eq!(samples[1].image.as_slice(), b"zero");
        assert_eq!(samples[2].image.as_slice(), b"two");
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.label)
                .collect::<Vec<_>>(),
            [2, 0, 2]
        );

        assert!(dataset.get_many(&[]).unwrap().is_empty());
        assert!(dataset.get_many(&[3]).is_err());
        assert_eq!(dataset.get(1).unwrap().label, 1);

        std::fs::remove_dir_all(path).ok();
    }

    #[test]
    fn rejects_non_native_image_or_label_schema() {
        // Schema validation is intentionally kept in a small helper test so
        // invalid datasets fail during construction, before get_many.
        let schema = Schema::new(vec![
            Field::new("image", DataType::Utf8, false),
            Field::new("label", DataType::Int32, false),
        ]);
        let error = validate_schema(&schema, "image", "label").unwrap_err();
        assert!(error.to_string().contains("binary"));
    }
}
