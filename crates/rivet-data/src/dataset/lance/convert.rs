//! Convert local tabular files into a Lance split bundle.

use crate::dataset::MANIFEST_FILE_NAME;
use crate::errors::{DataResult, invalid_argument};
use arrow::array::new_null_array;
use arrow::compute::cast;
use arrow::csv::reader::{Format, ReaderBuilder as CsvReaderBuilder};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::json::ReaderBuilder as JsonReaderBuilder;
use arrow::json::reader::{infer_json_schema_from_iterator, infer_json_schema_from_seekable};
use arrow::record_batch::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use lance::Dataset;
use lance::dataset::WriteParams;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Builder;

const BATCH_SIZE: usize = 4096;
type BatchReader = Box<dyn RecordBatchReader + Send>;

/// Convert selected local files and write a manifest to a fresh output directory.
/// The caller owns publication of the directory upon success.
pub fn convert_files_to_lance(
    output: &Path,
    name: &str,
    mut files: BTreeMap<String, Vec<PathBuf>>,
    json_columns: Option<&[String]>,
    detect_json_splits: bool,
) -> DataResult<()> {
    if files.is_empty() || files.values().any(Vec::is_empty) {
        return Err(invalid_argument(
            "each Lance split needs at least one input file",
        ));
    }
    fs::create_dir_all(output)?;
    let mut groups = BTreeMap::new();
    let single_json = detect_json_splits
        && files.len() == 1
        && files
            .values()
            .next()
            .is_some_and(|paths| paths.len() == 1 && extension(&paths[0]) == "json");

    if single_json {
        let (split, paths) = files.pop_first().unwrap();
        let value = read_json(&paths[0])?;
        match value {
            Value::Object(parts) if !parts.is_empty() && parts.values().all(Value::is_array) => {
                let mut empty = Vec::new();
                for (split, value) in parts {
                    let Value::Array(rows) = value else {
                        unreachable!()
                    };
                    if rows.is_empty() {
                        empty.push(split);
                    } else {
                        groups.insert(split, vec![json_rows(rows, json_columns)?]);
                    }
                }
                if !empty.is_empty() {
                    let schema =
                        merged_schema(groups.values().flatten().map(|reader| reader.schema()))?;
                    for split in empty {
                        let reader = RecordBatchIterator::new(std::iter::empty(), schema.clone());
                        groups.insert(split, vec![Box::new(reader)]);
                    }
                }
            }
            value => {
                groups.insert(split, vec![json_array(value, &paths[0], json_columns)?]);
            }
        }
    }

    let groups = groups
        .into_iter()
        .map(Ok)
        .chain(files.into_iter().map(|(split, paths)| {
            let readers = paths
                .iter()
                .map(|path| open_file(path, json_columns))
                .collect::<DataResult<Vec<_>>>()?;
            Ok::<_, crate::errors::DataError>((split, readers))
        }));
    let runtime = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let mut splits = BTreeMap::new();
    for group in groups {
        let (split, readers) = group?;
        if split.is_empty()
            || !split
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err(invalid_argument(format!(
                "invalid Lance split name: {split:?}"
            )));
        }
        let file_name = format!("{split}.lance");
        let path = output.join(&file_name);
        let uri = path
            .to_str()
            .ok_or_else(|| invalid_argument("Lance output path must be UTF-8"))?;
        let schema = merged_schema(readers.iter().map(|reader| reader.schema()))?;
        let target_schema = schema.clone();
        let batches = readers
            .into_iter()
            .flatten()
            .map(move |batch| align_batch(batch?, &target_schema));
        let params = WriteParams::with_storage_version("2.1".parse()?);
        let rows = runtime.block_on(async {
            let dataset =
                Dataset::write(RecordBatchIterator::new(batches, schema), uri, Some(params))
                    .await?;
            dataset.count_rows(None).await
        })?;
        splits.insert(split, json!({"path": file_name, "num_rows": rows}));
    }
    let manifest = json!({"format_version": 1, "name": name, "splits": splits});
    let manifest_file = File::create(output.join(MANIFEST_FILE_NAME))?;
    serde_json::to_writer_pretty(manifest_file, &manifest).map_err(std::io::Error::other)?;
    Ok(())
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn read_json(path: &Path) -> DataResult<Value> {
    let reader = BufReader::new(File::open(path)?);
    serde_json::from_reader(reader)
        .map_err(|err| invalid_argument(format!("{}: {err}", path.display())))
}

fn open_file(path: &Path, columns: Option<&[String]>) -> DataResult<BatchReader> {
    let suffix = extension(path);
    if suffix == "json" {
        return json_array(read_json(path)?, path, columns);
    }
    let mut file = BufReader::new(File::open(path)?);
    let reader: BatchReader = match suffix.as_str() {
        "parquet" => {
            let parquet_error = |err| ArrowError::ExternalError(Box::new(err));
            let reader = ParquetRecordBatchReaderBuilder::try_new(file.into_inner())
                .map_err(parquet_error)?;
            Box::new(
                reader
                    .with_batch_size(BATCH_SIZE)
                    .build()
                    .map_err(parquet_error)?,
            )
        }
        "arrow" => match FileReader::try_new(file, None) {
            Ok(reader) => Box::new(reader),
            Err(_) => Box::new(StreamReader::try_new_buffered(File::open(path)?, None)?),
        },
        "csv" | "tsv" => {
            let delimiter = if suffix == "tsv" { b'\t' } else { b',' };
            let format = Format::default()
                .with_header(true)
                .with_delimiter(delimiter);
            let (schema, _) = format.infer_schema(&mut file, None)?;
            file.rewind()?;
            Box::new(
                CsvReaderBuilder::new(Arc::new(schema))
                    .with_format(format)
                    .with_batch_size(BATCH_SIZE)
                    .build_buffered(file)?,
            )
        }
        "jsonl" | "ndjson" => {
            let (schema, _) = infer_json_schema_from_seekable(&mut file, None)?;
            Box::new(
                JsonReaderBuilder::new(Arc::new(schema))
                    .with_batch_size(BATCH_SIZE)
                    .build(file)?,
            )
        }
        _ => {
            return Err(invalid_argument(format!(
                "unsupported input format: {}",
                path.display()
            )));
        }
    };
    Ok(reader)
}

fn json_array(value: Value, path: &Path, columns: Option<&[String]>) -> DataResult<BatchReader> {
    let Value::Array(rows) = value else {
        return Err(invalid_argument(format!(
            "expected a JSON record array in {}",
            path.display()
        )));
    };
    json_rows(rows, columns)
}

fn json_rows(mut rows: Vec<Value>, columns: Option<&[String]>) -> DataResult<BatchReader> {
    let positional_names = if let Some(Value::Array(first)) = rows.first() {
        let default_names;
        let names = match columns {
            Some(columns) => columns,
            None => {
                default_names = (0..first.len())
                    .map(|i| format!("column_{i}"))
                    .collect::<Vec<_>>();
                &default_names
            }
        };
        for row in &mut rows {
            let Value::Array(values) = row else {
                return Err(invalid_argument(
                    "JSON data must contain object records or fixed-width positional records",
                ));
            };
            if values.len() != names.len() {
                return Err(invalid_argument(
                    "positional JSON records must match json_columns",
                ));
            }
            *row = Value::Object(names.iter().cloned().zip(std::mem::take(values)).collect());
        }
        Some(names.to_vec())
    } else {
        None
    };
    let inferred = infer_json_schema_from_iterator(rows.iter().map(Ok))?;
    let schema = match positional_names {
        Some(names) => {
            let order = names
                .iter()
                .map(|name| inferred.index_of(name))
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(inferred.project(&order)?)
        }
        None => Arc::new(inferred),
    };
    let mut decoder = JsonReaderBuilder::new(schema.clone())
        .with_batch_size(BATCH_SIZE)
        .build_decoder()?;
    let mut rows = rows.into_iter();
    let batches = std::iter::from_fn(move || {
        let chunk: Vec<_> = rows.by_ref().take(BATCH_SIZE).collect();
        if chunk.is_empty() {
            return None;
        }
        match decoder.serialize(&chunk).and_then(|()| decoder.flush()) {
            Ok(batch) => batch.map(Ok),
            Err(err) => Some(Err(err)),
        }
    });
    Ok(Box::new(RecordBatchIterator::new(batches, schema)))
}

fn merged_schema(schemas: impl IntoIterator<Item = SchemaRef>) -> DataResult<SchemaRef> {
    let schemas: Vec<_> = schemas.into_iter().collect();
    let mut types = BTreeMap::<_, DataType>::new();
    for field in schemas.iter().flat_map(|schema| schema.fields()) {
        types
            .entry(field.name())
            .and_modify(|data_type| {
                if let Some(promoted) = promote_type(data_type, field.data_type()) {
                    *data_type = promoted;
                }
            })
            .or_insert_with(|| field.data_type().clone());
    }
    let mut merged = Schema::try_merge(schemas.iter().map(|schema| {
        let fields = schema
            .fields()
            .iter()
            .map(|field| {
                let target = &types[field.name()];
                if field.data_type() != target && promote_type(field.data_type(), target).is_some()
                {
                    Arc::new(field.as_ref().clone().with_data_type(target.clone()))
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        Schema::new_with_metadata(fields, schema.metadata().clone())
    }))?;
    if merged.fields().is_empty() {
        return Err(invalid_argument(
            "cannot infer a Lance schema from empty input",
        ));
    }
    let fields: Vec<_> = merged
        .fields
        .iter()
        .map(|field| {
            if schemas
                .iter()
                .any(|schema| schema.index_of(field.name()).is_err())
            {
                Arc::new(field.as_ref().clone().with_nullable(true))
            } else {
                field.clone()
            }
        })
        .collect();
    merged.fields = fields.into();
    Ok(Arc::new(merged))
}

fn promote_type(left: &DataType, right: &DataType) -> Option<DataType> {
    use DataType::*;
    if left == right || right == &Null {
        return Some(left.clone());
    }
    if left == &Null {
        return Some(right.clone());
    }
    let numeric = [
        Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float16, Float32, Float64,
    ];
    let signed = [Int8, Int16, Int32, Int64];
    let unsigned = [UInt8, UInt16, UInt32, UInt64];
    let floats = [Float16, Float32, Float64];
    for widths in [signed.as_slice(), unsigned.as_slice(), floats.as_slice()] {
        if let (Some(a), Some(b)) = (
            widths.iter().position(|kind| kind == left),
            widths.iter().position(|kind| kind == right),
        ) {
            return Some(widths[a.max(b)].clone());
        }
    }
    if numeric.contains(left) && numeric.contains(right) {
        if floats.contains(left) || floats.contains(right) {
            return Some(Float64);
        }
        if left == &UInt64 || right == &UInt64 {
            return Some(Decimal128(20, 0));
        }
        return Some(Int64);
    }
    match (left, right) {
        (Utf8, LargeUtf8) | (LargeUtf8, Utf8) => Some(LargeUtf8),
        (Binary, LargeBinary) | (LargeBinary, Binary) => Some(LargeBinary),
        (Decimal128(20, 0), value) | (value, Decimal128(20, 0))
            if signed.contains(value) || unsigned.contains(value) =>
        {
            Some(Decimal128(20, 0))
        }
        _ => None,
    }
}

fn align_batch(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch, ArrowError> {
    if batch.schema() == *schema {
        return Ok(batch);
    }
    let columns = schema
        .fields()
        .iter()
        .map(|field| match batch.column_by_name(field.name()) {
            Some(column) if column.data_type() == field.data_type() => Ok(column.clone()),
            Some(column) => cast(column, field.data_type()),
            None => Ok(new_null_array(field.data_type(), batch.num_rows())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema.clone(), columns)
}
