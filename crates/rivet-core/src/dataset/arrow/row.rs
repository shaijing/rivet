use crate::errors::{RivetResult, invalid_argument};
use arrow::array::{
    Array, BinaryArray, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray, StringArray,
    StructArray,
};
use arrow::record_batch::RecordBatch;
use arrow_buffer::Buffer;

/// A single row view over an Arrow table with typed, zero-copy column
/// extraction. Extraction understands Arrow types, not modalities.
pub struct ArrowRow<'a> {
    batch: &'a RecordBatch,
    row: usize,
}

impl<'a> ArrowRow<'a> {
    pub(crate) fn new(batch: &'a RecordBatch, row: usize) -> Self {
        Self { batch, row }
    }

    fn column(&self, column: &str) -> RivetResult<&arrow::array::ArrayRef> {
        self.batch
            .column_by_name(column)
            .ok_or_else(|| invalid_argument(format!("missing column {column}")))
    }

    /// Zero-copy slice of a binary column's values buffer.
    pub fn binary(&self, column: &str) -> RivetResult<Buffer> {
        let array = self.column(column)?;

        if let Some(bytes) = array.as_any().downcast_ref::<BinaryArray>() {
            if bytes.is_null(self.row) {
                return Err(null_value(column, self.row));
            }
            let offsets = bytes.value_offsets();
            return Ok(slice(bytes.values(), offsets, self.row));
        }

        if let Some(bytes) = array.as_any().downcast_ref::<LargeBinaryArray>() {
            if bytes.is_null(self.row) {
                return Err(null_value(column, self.row));
            }
            let offsets = bytes.value_offsets();
            return Ok(slice(bytes.values(), offsets, self.row));
        }

        Err(invalid_argument(format!(
            "{column} must be binary or large_binary, got {:?}",
            array.data_type()
        )))
    }

    /// Zero-copy slice of a struct column's binary field (e.g. the Hugging
    /// Face `img: struct<bytes: binary>` layout).
    pub fn struct_binary(&self, struct_column: &str, field: &str) -> RivetResult<Buffer> {
        let struct_col = self.column(struct_column)?;
        let image = struct_col
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                invalid_argument(format!(
                    "{struct_column} must be a struct column, got {:?}",
                    struct_col.data_type()
                ))
            })?;
        let bytes = image
            .column_by_name(field)
            .ok_or_else(|| invalid_argument(format!("{struct_column}.{field} field is missing")))?;

        let label = format!("{struct_column}.{field}");
        if let Some(bytes) = bytes.as_any().downcast_ref::<BinaryArray>() {
            if bytes.is_null(self.row) {
                return Err(null_value(&label, self.row));
            }
            let offsets = bytes.value_offsets();
            return Ok(slice(bytes.values(), offsets, self.row));
        }

        if let Some(bytes) = bytes.as_any().downcast_ref::<LargeBinaryArray>() {
            if bytes.is_null(self.row) {
                return Err(null_value(&label, self.row));
            }
            let offsets = bytes.value_offsets();
            return Ok(slice(bytes.values(), offsets, self.row));
        }

        Err(invalid_argument(format!(
            "{label} must be binary or large_binary, got {:?}",
            bytes.data_type()
        )))
    }

    /// Zero-copy slice of a UTF-8 column's values buffer (StringArray or
    /// LargeStringArray).
    pub fn utf8(&self, column: &str) -> RivetResult<Buffer> {
        let array = self.column(column)?;

        if let Some(text) = array.as_any().downcast_ref::<StringArray>() {
            if text.is_null(self.row) {
                return Err(null_value(column, self.row));
            }
            let offsets = text.value_offsets();
            return Ok(slice(text.values(), offsets, self.row));
        }

        if let Some(text) = array.as_any().downcast_ref::<LargeStringArray>() {
            if text.is_null(self.row) {
                return Err(null_value(column, self.row));
            }
            let offsets = text.value_offsets();
            return Ok(slice(text.values(), offsets, self.row));
        }

        Err(invalid_argument(format!(
            "{column} must be utf8 or large_utf8, got {:?}",
            array.data_type()
        )))
    }

    /// Read an integer column as `i64`; null rows are an error. Accepts
    /// int64 or int32 columns.
    pub fn i64(&self, column: &str) -> RivetResult<i64> {
        self.optional_i64(column)?
            .ok_or_else(|| null_value(column, self.row))
    }

    /// Read an integer column as `Option<i64>`; null rows yield `None`.
    pub fn optional_i64(&self, column: &str) -> RivetResult<Option<i64>> {
        let array = self.column(column)?;

        if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
            return Ok(if values.is_null(self.row) {
                None
            } else {
                Some(values.value(self.row))
            });
        }

        if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
            return Ok(if values.is_null(self.row) {
                None
            } else {
                Some(i64::from(values.value(self.row)))
            });
        }

        Err(invalid_argument(format!(
            "{column} must be int64 or int32, got {:?}",
            array.data_type()
        )))
    }
}

/// Row `row`'s payload is `[offsets[row], offsets[row + 1])` inside the
/// shared values buffer; slicing shares the backing allocation.
fn slice<O: SliceOffset>(values: &Buffer, offsets: &[O], row: usize) -> Buffer {
    let start = offsets[row].to_usize();
    let end = offsets[row + 1].to_usize();
    values.slice_with_length(start, end - start)
}

trait SliceOffset: Copy {
    fn to_usize(self) -> usize;
}

impl SliceOffset for i32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl SliceOffset for i64 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

fn null_value(column: &str, row: usize) -> crate::errors::RivetError {
    invalid_argument(format!("{column} is null at row {row}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ArrayRef;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn build() -> RecordBatch {
        let bytes_field = Arc::new(Field::new("bytes", DataType::Binary, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("img", DataType::Struct(vec![bytes_field.clone()].into()), true),
            Field::new("raw", DataType::Binary, true),
            Field::new("text", DataType::Utf8, true),
            Field::new("label", DataType::Int64, false),
            Field::new("opt", DataType::Int32, true),
        ]));

        let img_bytes: BinaryArray = [Some(&b"one"[..]), Some(&b"two"[..]), None]
            .into_iter()
            .collect();
        let raw: BinaryArray = [Some(&b"a"[..]), Some(&b"bc"[..]), Some(&b""[..])]
            .into_iter()
            .collect();
        let text: StringArray = [Some("hello"), Some(""), Some("wörld")].into_iter().collect();
        let label: Int64Array = [Some(7), Some(8), Some(9)].into_iter().collect();
        let opt: Int32Array = [Some(1), None, Some(3)].into_iter().collect();

        let img_bytes_ref: ArrayRef = Arc::new(img_bytes);
        let img = StructArray::from(vec![(bytes_field, img_bytes_ref)]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(img),
                Arc::new(raw),
                Arc::new(text),
                Arc::new(label),
                Arc::new(opt),
            ],
        )
        .unwrap()
    }

    /// Byte range of the values buffer that holds a column's payloads.
    fn values_bounds(batch: &RecordBatch, column: &str) -> (usize, usize) {
        let data = batch.column_by_name(column).unwrap().to_data();
        let buffers = data.buffers();
        let values = buffers.last().unwrap();
        let start = values.as_ptr() as usize;
        (start, start + values.len())
    }

    #[test]
    fn typed_accessors_are_zero_copy() {
        let batch = build();
        let (raw_start, raw_end) = values_bounds(&batch, "raw");
        let (text_start, text_end) = values_bounds(&batch, "text");

        let row = ArrowRow::new(&batch, 0);
        let image = row.struct_binary("img", "bytes").unwrap();
        assert_eq!(image.as_slice(), b"one");
        let raw = row.binary("raw").unwrap();
        assert_eq!(raw.as_slice(), b"a");
        let text = row.utf8("text").unwrap();
        assert_eq!(text.as_slice(), b"hello");
        assert_eq!(row.i64("label").unwrap(), 7);
        assert_eq!(row.optional_i64("opt").unwrap(), Some(1));

        let raw_ptr = raw.as_ptr() as usize;
        assert!(raw_ptr >= raw_start && raw_ptr + raw.len() <= raw_end);
        let text_ptr = text.as_ptr() as usize;
        assert!(text_ptr >= text_start && text_ptr + text.len() <= text_end);
    }

    #[test]
    fn nulls_and_optional_labels() {
        let batch = build();
        assert!(ArrowRow::new(&batch, 2).struct_binary("img", "bytes").is_err());
        assert_eq!(
            ArrowRow::new(&batch, 1).optional_i64("opt").unwrap(),
            None
        );
        assert!(ArrowRow::new(&batch, 1).i64("opt").is_err());
        assert_eq!(ArrowRow::new(&batch, 0).optional_i64("label").unwrap(), Some(7));
    }

    #[test]
    fn wrong_type_and_missing_columns() {
        let batch = build();
        assert!(ArrowRow::new(&batch, 0).utf8("raw").is_err());
        assert!(ArrowRow::new(&batch, 0).binary("text").is_err());
        assert!(ArrowRow::new(&batch, 0).i64("text").is_err());
        assert!(ArrowRow::new(&batch, 0).binary("nope").is_err());
        assert!(ArrowRow::new(&batch, 0).utf8("nope").is_err());
    }

    #[test]
    fn empty_and_wide_values_slice_correctly() {
        let batch = build();
        let text = ArrowRow::new(&batch, 1).utf8("text").unwrap();
        assert_eq!(text.as_slice(), b"");
        let text = ArrowRow::new(&batch, 2).utf8("text").unwrap();
        assert_eq!(text.as_slice(), "wörld".as_bytes());
        let raw = ArrowRow::new(&batch, 2).binary("raw").unwrap();
        assert_eq!(raw.as_slice(), b"");
    }
}
