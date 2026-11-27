use super::row::read_encoded_sample;
use super::schema::validate_schema;
use crate::sample::image::EncodedImageSample;
use rivet_data::dataset::lance::LanceTable;
use rivet_data::dataset::source::{AccessPattern, Dataset, SourceCapabilities, validate_indices};
use rivet_data::errors::{DataResult, invalid_argument};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Rivet's native Lance image classification dataset.
pub struct LanceImageDataset {
    path: PathBuf,
    pub(super) table: Arc<LanceTable>,
    pub(super) image_column: String,
    pub(super) label_column: String,
}

impl LanceImageDataset {
    pub fn open(
        path: impl AsRef<Path>,
        image_column: impl Into<String>,
        label_column: impl Into<String>,
    ) -> DataResult<Self> {
        Self::open_with_image_alias(path, image_column, label_column, false)
    }

    pub(crate) fn open_with_image_alias(
        path: impl AsRef<Path>,
        image_column: impl Into<String>,
        label_column: impl Into<String>,
        allow_img: bool,
    ) -> DataResult<Self> {
        let path = path.as_ref().to_owned();
        let table = Arc::new(LanceTable::open(&path)?);
        let mut image_column = image_column.into();
        if allow_img
            && image_column == "image"
            && table.schema().index_of("image").is_err()
            && table.schema().index_of("img").is_ok()
        {
            image_column = "img".into();
        }
        let label_column = label_column.into();
        validate_schema(table.schema(), &image_column, &label_column)?;

        Ok(Self {
            path,
            table,
            image_column,
            label_column,
        })
    }

    pub fn new(
        path: impl AsRef<Path>,
        image_column: impl Into<String>,
        label_column: impl Into<String>,
    ) -> crate::errors::RivetResult<Self> {
        Ok(Self::open(path, image_column, label_column)?)
    }

    pub fn open_default(path: impl AsRef<Path>) -> crate::errors::RivetResult<Self> {
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
}

impl Dataset for LanceImageDataset {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn source_path(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            access_pattern: AccessPattern::RandomAccess,
            batched_reads: true,
            preferred_batch_size: None,
            zero_copy: false,
            parallel_reads: false,
            async_reads: false,
            read_device: Some("cpu"),
        }
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
            .map(|row| {
                read_encoded_sample(
                    self.table.schema(),
                    &self.image_column,
                    &self.label_column,
                    &batch,
                    row,
                )
            })
            .collect()
    }
}
