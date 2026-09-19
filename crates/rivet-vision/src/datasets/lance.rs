use crate::errors::{RivetResult, invalid_argument as vision_invalid_argument};
use crate::sample::image::EncodedImageSample;
#[cfg(test)]
use crate::sample::image::ImageSample;
use arrow::datatypes::{DataType, Schema};
use rivet_data::dataset::arrow::ArrowRow;
use rivet_data::dataset::lance::LanceTable;
use rivet_data::dataset::source::{Dataset, validate_indices};
use rivet_data::dataset::{DatasetBundle, DatasetLoadResult, DatasetManifest, MANIFEST_FILE_NAME};
use rivet_data::errors::{DataResult, invalid_argument};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

/// Rivet's native Lance image classification dataset.
///
/// The hot-path schema is intentionally small: image binary or large_binary,
/// and label int32 or int64. Additional Lance columns are ignored through
/// projection.
/// A configured Hugging Face-compatible struct<bytes: binary> image column is
/// accepted as a compatibility input, but is not Rivet's native schema.
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
    ) -> DataResult<Self> {
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
        Ok(Self::open(path, image_column, label_column)?)
    }

    pub fn open_default(path: impl AsRef<Path>) -> RivetResult<Self> {
        Ok(Self::open(path, "image", "label")?)
    }

    pub fn image_column(&self) -> &str {
        &self.image_column
    }

    pub fn label_column(&self) -> &str {
        &self.label_column
    }

    pub fn schema(&self) -> &arrow::datatypes::SchemaRef {
        self.table.schema()
    }

    fn get_one(
        &self,
        batch: &arrow_array::RecordBatch,
        row: usize,
    ) -> DataResult<EncodedImageSample> {
        let row = ArrowRow::new(batch, row);
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
            // ArrowRow slices the BinaryArray values buffer, so this keeps
            // Lance's encoded payload allocation alive without copying bytes.
            image,
            label: row.i64(&self.label_column)?,
        })
    }
}

impl Dataset for LanceImageDataset {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
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

fn validate_schema(schema: &Schema, image_column: &str, label_column: &str) -> DataResult<()> {
    let image = schema
        .field_with_name(image_column)
        .map_err(invalid_argument)?;
    let valid_image = match image.data_type() {
        DataType::Binary | DataType::LargeBinary => true,
        DataType::Struct(fields) => fields
            .iter()
            .find(|field| field.name() == "bytes")
            .is_some_and(|field| {
                matches!(field.data_type(), DataType::Binary | DataType::LargeBinary)
            }),
        _ => false,
    };
    if !valid_image {
        return Err(invalid_argument(format!(
            "{image_column} must be binary, large_binary, or a struct with a binary bytes field, got {:?}",
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

/// Open a physical Lance dataset or a logical root containing split datasets.
pub fn load_lance_image_dataset(
    path: impl AsRef<Path>,
    image_column: impl Into<String>,
    label_column: impl Into<String>,
) -> RivetResult<DatasetLoadResult<crate::source::ImageSource>> {
    let path = path.as_ref();
    let image_column = image_column.into();
    let label_column = label_column.into();

    if is_lance_path(path) {
        let dataset = Arc::new(LanceImageDataset::open(path, image_column, label_column)?);
        return Ok(DatasetLoadResult::Single(
            crate::source::ImageSource::from_encoded(dataset),
        ));
    }
    if !path.exists() {
        return Err(vision_invalid_argument(format!(
            "dataset root does not exist: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(vision_invalid_argument(format!(
            "dataset root is not a directory: {}",
            path.display()
        )));
    }

    let split_paths = if path.join(MANIFEST_FILE_NAME).is_file() {
        manifest_split_paths(path)?
    } else {
        discover_split_paths(path)?
    };

    let mut splits = BTreeMap::new();
    for (name, split_path, expected_rows) in split_paths {
        let dataset = Arc::new(LanceImageDataset::open(
            &split_path,
            image_column.clone(),
            label_column.clone(),
        )?);
        if let Some(expected_rows) = expected_rows {
            if dataset.len() != expected_rows {
                return Err(vision_invalid_argument(format!(
                    "manifest row count for split '{name}' is {expected_rows}, but Lance reports {}",
                    dataset.len()
                )));
            }
        }
        splits.insert(name, crate::source::ImageSource::from_encoded(dataset));
    }

    Ok(DatasetLoadResult::Bundle(DatasetBundle::from_splits(
        splits,
    )))
}

fn is_lance_path(path: &Path) -> bool {
    path.extension() == Some(OsStr::new("lance"))
}

fn manifest_split_paths(root: &Path) -> RivetResult<Vec<(String, PathBuf, Option<usize>)>> {
    let manifest = DatasetManifest::read(root)?;
    manifest
        .splits
        .iter()
        .map(|(name, split)| {
            Ok((
                name.clone(),
                manifest.split_path(root, name, split)?,
                split.num_rows,
            ))
        })
        .collect()
}

fn discover_split_paths(root: &Path) -> DataResult<Vec<(String, PathBuf, Option<usize>)>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || !is_lance_path(&path) {
            continue;
        }
        let Some(name) = path.file_stem().and_then(OsStr::to_str) else {
            continue;
        };
        paths.push((name.to_owned(), path, None));
    }
    paths.sort_by(|left, right| left.0.cmp(&right.0));

    if paths.is_empty() {
        return Err(invalid_argument(format!(
            "dataset root '{}' contains no .lance split directories and no {}",
            root.display(),
            MANIFEST_FILE_NAME
        )));
    }
    Ok(paths)
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
    fn encoded_cache_matches_the_lazy_source() {
        let path = temp_lance_path();
        write_fixture(&path);

        let source = crate::source::ImageSource::from_encoded(Arc::new(
            LanceImageDataset::open_default(&path).unwrap(),
        ));
        let expected = source.get_many(&[2, 0, 2]).unwrap();
        let cached = source.cache_encoded(2).unwrap();
        let actual = cached.get_many(&[2, 0, 2]).unwrap();

        assert_eq!(
            actual
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.label,
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.label,
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.image.as_slice(),
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.image.as_slice(),
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>()
        );

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

    #[test]
    fn accepts_huggingface_struct_image_as_compatibility_input() {
        let schema = Schema::new(vec![
            Field::new(
                "img",
                DataType::Struct(
                    vec![Arc::new(Field::new("bytes", DataType::Binary, true))].into(),
                ),
                true,
            ),
            Field::new("label", DataType::Int64, false),
        ]);
        validate_schema(&schema, "img", "label").unwrap();
    }
}
