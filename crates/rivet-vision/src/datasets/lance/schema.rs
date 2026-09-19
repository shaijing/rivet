use arrow::datatypes::{DataType, Schema};
use rivet_data::errors::{DataResult, invalid_argument};

pub(crate) fn validate_schema(
    schema: &Schema,
    image_column: &str,
    label_column: &str,
) -> DataResult<()> {
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
