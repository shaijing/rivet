use crate::sample::image::EncodedImageSample;
use arrow::datatypes::{DataType, Schema};
use rivet_data::dataset::source::validate_indices;
use rivet_data::dataset::{Dataset, MmapArrowTable};
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
}

impl Dataset for ArrowImageDataset {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.table.len()
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        validate_indices(indices, self.len())?;
        indices.iter().map(|&index| self.get_one(index)).collect()
    }
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
