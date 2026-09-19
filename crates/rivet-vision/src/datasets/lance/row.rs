use crate::sample::image::EncodedImageSample;
use arrow::datatypes::{DataType, Schema};
use rivet_data::dataset::arrow::ArrowRow;
use rivet_data::errors::{DataResult, invalid_argument};

pub(super) fn read_encoded_sample(
    schema: &Schema,
    image_column: &str,
    label_column: &str,
    batch: &arrow_array::RecordBatch,
    row: usize,
) -> DataResult<EncodedImageSample> {
    let row = ArrowRow::new(batch, row);
    let image = match schema
        .field_with_name(image_column)
        .map_err(invalid_argument)?
        .data_type()
    {
        DataType::Binary | DataType::LargeBinary => row.binary(image_column)?,
        DataType::Struct(_) => row.struct_binary(image_column, "bytes")?,
        data_type => {
            return Err(invalid_argument(format!(
                "{} must be binary, large_binary, or a struct with a bytes field, got {:?}",
                image_column, data_type
            )));
        }
    };
    Ok(EncodedImageSample {
        // ArrowRow slices the BinaryArray values buffer, so this keeps
        // Lance's encoded payload allocation alive without copying bytes.
        image,
        label: row.i64(label_column)?,
    })
}
