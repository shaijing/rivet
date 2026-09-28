//! Convert local Hugging Face Arrow IPC caches into Rivet-native Lance data.
//!
//! Hugging Face's datasets library stores downloaded splits as Arrow IPC
//! files. This module accepts those local files rather than depending on
//! Python or on a Hub client. The conversion is streaming: each input record
//! batch is normalized and handed directly to Lance.

use arrow::array::{Array, ArrayRef, StructArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use arrow_ipc::reader::StreamReader;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Builder;
use walkdir::WalkDir;

use crate::errors::{RivetResult, invalid_argument};

/// One Hugging Face split and all Arrow IPC shards belonging to it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuggingFaceSplit {
    pub name: String,
    pub arrow_files: Vec<PathBuf>,
}

impl HuggingFaceSplit {
    pub fn new(name: impl Into<String>, arrow_files: Vec<PathBuf>) -> Self {
        Self {
            name: name.into(),
            arrow_files,
        }
    }
}

/// Options for converting Hugging Face image classification data.
#[derive(Clone, Debug)]
pub struct HuggingFaceLanceOptions {
    /// Hugging Face image feature column, usually img.
    pub image_column: String,
    /// Integer class label column, usually label.
    pub label_column: String,
    /// Preserve img.path as an optional native path column.
    pub keep_path: bool,
    /// Replace an existing output root.
    pub overwrite: bool,
    /// Maximum physical Lance data-file size in bytes.
    ///
    /// Lance checks this limit after each written row group, so the final
    /// file can be slightly larger than this value. The default is 2 GiB.
    pub max_bytes_per_file: usize,
    /// Maximum number of rows in one physical Lance data file.
    pub max_rows_per_file: usize,
}

pub const DEFAULT_MAX_BYTES_PER_FILE: usize = 2 * 1024 * 1024 * 1024;
pub const DEFAULT_MAX_ROWS_PER_FILE: usize = 1024 * 1024;

impl Default for HuggingFaceLanceOptions {
    fn default() -> Self {
        Self {
            image_column: "img".to_owned(),
            label_column: "label".to_owned(),
            keep_path: false,
            overwrite: false,
            max_bytes_per_file: DEFAULT_MAX_BYTES_PER_FILE,
            max_rows_per_file: DEFAULT_MAX_ROWS_PER_FILE,
        }
    }
}

/// Result of a completed conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LanceConversionReport {
    pub output_root: PathBuf,
    pub rows_by_split: BTreeMap<String, usize>,
}

/// Discover common Hugging Face Arrow cache layouts below root.
///
/// Recognizes filenames such as train-00000-of-00001.arrow,
/// cifar10-train.arrow, and validation.arrow. If root is itself an Arrow
/// file, it is treated as the train split. For unusual split names, construct
/// HuggingFaceSplit explicitly and call convert_huggingface_dataset.
pub fn discover_huggingface_splits(root: impl AsRef<Path>) -> RivetResult<Vec<HuggingFaceSplit>> {
    let root = root.as_ref();
    if root.is_file() {
        if root.extension().and_then(|ext| ext.to_str()) != Some("arrow") {
            return Err(invalid_argument(format!(
                "Hugging Face input file must have an .arrow extension: {}",
                root.display()
            )));
        }
        return Ok(vec![HuggingFaceSplit::new(
            "train",
            vec![root.to_path_buf()],
        )]);
    }
    if !root.is_dir() {
        return Err(invalid_argument(format!(
            "Hugging Face input root does not exist or is not a directory: {}",
            root.display()
        )));
    }

    let mut grouped = BTreeMap::<String, Vec<PathBuf>>::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.map_err(|error| invalid_argument(error.to_string()))?;
        let path = entry.path();
        if !entry.file_type().is_file()
            || path.extension().and_then(|ext| ext.to_str()) != Some("arrow")
        {
            continue;
        }
        let Some(split) = infer_split_name(path) else {
            continue;
        };
        grouped.entry(split).or_default().push(path.to_path_buf());
    }

    if grouped.is_empty() {
        return Err(invalid_argument(format!(
            "no recognizable Hugging Face split Arrow files found below {}",
            root.display()
        )));
    }

    Ok(grouped
        .into_iter()
        .map(|(name, mut arrow_files)| {
            arrow_files.sort();
            HuggingFaceSplit::new(name, arrow_files)
        })
        .collect())
}

/// Convert all supplied Hugging Face Arrow shards into a Rivet dataset root.
///
/// Each split is written as output_root/<split>.lance, followed by a v2
/// dataset.json manifest. The image payload is normalized to non-null binary
/// and the label to non-null int32, Rivet's native encoded image format.
pub fn convert_huggingface_dataset(
    splits: impl IntoIterator<Item = HuggingFaceSplit>,
    output_root: impl AsRef<Path>,
    options: &HuggingFaceLanceOptions,
) -> RivetResult<LanceConversionReport> {
    if options.max_bytes_per_file == 0 || options.max_rows_per_file == 0 {
        return Err(invalid_argument(
            "max_bytes_per_file and max_rows_per_file must be greater than zero",
        ));
    }
    let output_root = output_root.as_ref().to_path_buf();
    let mut split_map = BTreeMap::new();
    for split in splits {
        validate_split_name(&split.name)?;
        if split.arrow_files.is_empty() {
            return Err(invalid_argument(format!(
                "Hugging Face split '{}' has no Arrow files",
                split.name
            )));
        }
        if split_map.insert(split.name.clone(), split).is_some() {
            return Err(invalid_argument("duplicate Hugging Face split name"));
        }
    }
    if split_map.is_empty() {
        return Err(invalid_argument(
            "at least one Hugging Face split is required",
        ));
    }

    prepare_output_root(&output_root, options.overwrite)?;
    std::fs::create_dir_all(&output_root)
        .map_err(|error| invalid_argument(format!("create output root: {error}")))?;

    let schema = native_schema(options.keep_path);
    let mut rows_by_split = BTreeMap::new();
    for (name, split) in &split_map {
        let output_path = output_root.join(format!("{name}.lance"));
        let rows = convert_split(split, &output_path, options, schema.clone())?;
        rows_by_split.insert(name.clone(), rows);
    }

    write_manifest(&output_root, &rows_by_split, options.keep_path)?;
    Ok(LanceConversionReport {
        output_root,
        rows_by_split,
    })
}

fn convert_split(
    split: &HuggingFaceSplit,
    output_path: &Path,
    options: &HuggingFaceLanceOptions,
    schema: SchemaRef,
) -> RivetResult<usize> {
    let mut readers = Vec::with_capacity(split.arrow_files.len());
    let mut input_schema: Option<SchemaRef> = None;
    for path in &split.arrow_files {
        let file = File::open(path).map_err(|error| {
            invalid_argument(format!(
                "open Hugging Face Arrow file {}: {error}",
                path.display()
            ))
        })?;
        let reader = StreamReader::try_new_buffered(file, None).map_err(|error| {
            invalid_argument(format!(
                "read Hugging Face Arrow schema {}: {error}",
                path.display()
            ))
        })?;
        if let Some(expected) = &input_schema {
            if expected.as_ref() != reader.schema().as_ref() {
                return Err(invalid_argument(format!(
                    "Arrow schema mismatch in split '{}' at {}",
                    split.name,
                    path.display()
                )));
            }
        } else {
            validate_input_schema(&reader.schema(), options)?;
            input_schema = Some(reader.schema());
        }
        readers.push(reader);
    }

    let batch_reader = RecordBatchIterator::new(
        HuggingFaceBatchReader {
            readers,
            current: 0,
            options: options.clone(),
        },
        schema,
    );
    let output_uri = output_path.to_str().ok_or_else(|| {
        invalid_argument(format!(
            "output path is not valid UTF-8: {}",
            output_path.display()
        ))
    })?;
    let runtime = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|error| invalid_argument(format!("create Lance runtime: {error}")))?;
    let mut write_params = lance::dataset::WriteParams::default();
    write_params.max_bytes_per_file = options.max_bytes_per_file;
    write_params.max_rows_per_file = options.max_rows_per_file;
    let dataset = runtime
        .block_on(lance::Dataset::write(
            batch_reader,
            output_uri,
            Some(write_params),
        ))
        .map_err(|error| invalid_argument(format!("write Lance split {output_uri}: {error}")))?;
    runtime
        .block_on(dataset.count_rows(None))
        .map_err(|error| invalid_argument(format!("count Lance rows {output_uri}: {error}")))
}

struct HuggingFaceBatchReader {
    readers: Vec<StreamReader<BufReader<File>>>,
    current: usize,
    options: HuggingFaceLanceOptions,
}

impl Iterator for HuggingFaceBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let reader = self.readers.get_mut(self.current)?;
            match reader.next() {
                Some(Ok(batch)) => return Some(convert_batch(&batch, &self.options)),
                Some(Err(error)) => return Some(Err(error)),
                None => self.current += 1,
            }
        }
    }
}

fn convert_batch(
    batch: &RecordBatch,
    options: &HuggingFaceLanceOptions,
) -> Result<RecordBatch, ArrowError> {
    let image = image_bytes(batch, &options.image_column)?;
    let label = label_values(batch, &options.label_column)?;
    let mut columns = vec![image, label];
    if options.keep_path {
        columns.push(image_path(batch, &options.image_column)?);
    }
    RecordBatch::try_new(native_schema(options.keep_path), columns)
}

fn native_schema(keep_path: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("image", DataType::Binary, false),
        Field::new("label", DataType::Int32, false),
    ];
    if keep_path {
        fields.push(Field::new("path", DataType::Utf8, true));
    }
    Arc::new(Schema::new(fields))
}

fn validate_input_schema(
    schema: &SchemaRef,
    options: &HuggingFaceLanceOptions,
) -> Result<(), crate::errors::VisionError> {
    let image = schema
        .field_with_name(&options.image_column)
        .map_err(|error| invalid_argument(error.to_string()))?;
    let image_type = match image.data_type() {
        DataType::Binary | DataType::LargeBinary => image.data_type(),
        DataType::Struct(fields) => fields
            .iter()
            .find(|field| field.name() == "bytes")
            .map(|field| field.data_type())
            .ok_or_else(|| {
                invalid_argument(format!(
                    "{} struct has no bytes field",
                    options.image_column
                ))
            })?,
        data_type => {
            return Err(invalid_argument(format!(
                "{} must be binary, large_binary, or a struct with bytes, got {data_type:?}",
                options.image_column
            )));
        }
    };
    if !matches!(image_type, DataType::Binary | DataType::LargeBinary) {
        return Err(invalid_argument(format!(
            "{}.bytes must be binary or large_binary, got {image_type:?}",
            options.image_column
        )));
    }
    let label = schema
        .field_with_name(&options.label_column)
        .map_err(|error| invalid_argument(error.to_string()))?;
    if !is_integer(label.data_type()) {
        return Err(invalid_argument(format!(
            "{} must be an integer column, got {:?}",
            options.label_column,
            label.data_type()
        )));
    }
    if options.keep_path {
        let DataType::Struct(fields) = image.data_type() else {
            return Err(invalid_argument(
                "keep_path requires an image struct with a path field",
            ));
        };
        let path_type = fields
            .iter()
            .find(|field| field.name() == "path")
            .map(|field| field.data_type())
            .ok_or_else(|| invalid_argument("image struct has no path field"))?;
        if !matches!(path_type, DataType::Utf8 | DataType::LargeUtf8) {
            return Err(invalid_argument(format!(
                "{}.path must be utf8 or large_utf8, got {path_type:?}",
                options.image_column
            )));
        }
    }
    Ok(())
}

fn image_bytes(batch: &RecordBatch, column: &str) -> Result<ArrayRef, ArrowError> {
    let input = batch.column_by_name(column).ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!("missing image column {column}"))
    })?;
    let values = match input.data_type() {
        DataType::Binary => input.clone(),
        DataType::LargeBinary => cast(input.as_ref(), &DataType::Binary)?,
        DataType::Struct(_) => {
            if input.null_count() != 0 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{column} contains null values"
                )));
            }
            let image = input
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!("{column} is not a struct"))
                })?;
            let bytes = image.column_by_name("bytes").ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!("{column} has no bytes field"))
            })?;
            match bytes.data_type() {
                DataType::Binary => bytes.clone(),
                DataType::LargeBinary => cast(bytes.as_ref(), &DataType::Binary)?,
                data_type => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "{column}.bytes must be binary or large_binary, got {data_type:?}"
                    )));
                }
            }
        }
        data_type => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "{column} must be binary, large_binary, or a struct with bytes, got {data_type:?}"
            )));
        }
    };
    reject_nulls(values.as_ref(), column)?;
    Ok(values)
}

fn image_path(batch: &RecordBatch, column: &str) -> Result<ArrayRef, ArrowError> {
    let input = batch.column_by_name(column).ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!("missing image column {column}"))
    })?;
    let image = input
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("{column} must be a struct to keep its path"))
        })?;
    let path = image
        .column_by_name("path")
        .ok_or_else(|| ArrowError::InvalidArgumentError(format!("{column} has no path field")))?;
    match path.data_type() {
        DataType::Utf8 => Ok(path.clone()),
        DataType::LargeUtf8 => cast(path.as_ref(), &DataType::Utf8),
        data_type => Err(ArrowError::InvalidArgumentError(format!(
            "{column}.path must be utf8 or large_utf8, got {data_type:?}"
        ))),
    }
}

fn label_values(batch: &RecordBatch, column: &str) -> Result<ArrayRef, ArrowError> {
    let input = batch.column_by_name(column).ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!("missing label column {column}"))
    })?;
    if !is_integer(input.data_type()) {
        return Err(ArrowError::InvalidArgumentError(format!(
            "{column} must be an integer column, got {:?}",
            input.data_type()
        )));
    }
    let values = cast(input.as_ref(), &DataType::Int32)?;
    reject_nulls(values.as_ref(), column)?;
    Ok(values)
}

fn reject_nulls(array: &dyn Array, column: &str) -> Result<(), ArrowError> {
    if array.null_count() != 0 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "{column} contains null values"
        )));
    }
    Ok(())
}

fn is_integer(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn infer_split_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
    for split in ["train", "validation", "test", "dev"] {
        if stem == split
            || stem.starts_with(&format!("{split}-"))
            || stem.ends_with(&format!("-{split}"))
            || stem.contains(&format!("-{split}-"))
        {
            return Some(split.to_owned());
        }
    }
    None
}

fn validate_split_name(name: &str) -> RivetResult<()> {
    let path = Path::new(name);
    if name.is_empty()
        || path.is_absolute()
        || path.components().count() != 1
        || path.file_name().and_then(|value| value.to_str()) != Some(name)
    {
        return Err(invalid_argument(format!(
            "invalid split name {name:?}; split names must be one relative path component"
        )));
    }
    Ok(())
}

fn prepare_output_root(path: &Path, overwrite: bool) -> RivetResult<()> {
    if !path.exists() {
        return Ok(());
    }
    if !overwrite {
        return Err(invalid_argument(format!(
            "output already exists: {}; enable overwrite to replace it",
            path.display()
        )));
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
    .map_err(|error| invalid_argument(format!("remove existing output: {error}")))?;
    Ok(())
}

fn write_manifest(
    output_root: &Path,
    rows_by_split: &BTreeMap<String, usize>,
    keep_path: bool,
) -> RivetResult<()> {
    let mut features = serde_json::json!({
        "image": {
            "type": "image",
            "column": "image",
            "representation": "encoded"
        },
        "label": {
            "type": "class_label",
            "column": "label",
            "dtype": "int32"
        }
    });
    if keep_path {
        features["path"] = serde_json::json!({
            "type": "text",
            "column": "path"
        });
    }

    let splits = rows_by_split
        .iter()
        .map(|(name, rows)| {
            (
                name.clone(),
                serde_json::json!({
                    "uri": format!("{name}.lance"),
                    "num_rows": rows
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let dataset_name = output_root
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("rivet-dataset");
    let manifest = serde_json::json!({
        "format_version": 2,
        "dataset": {"name": dataset_name, "modality": "image"},
        "features": features,
        "splits": splits,
        "created_by": {
            "tool": "rivet",
            "version": env!("CARGO_PKG_VERSION"),
            "source_format": "huggingface-arrow"
        }
    });
    let contents = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| invalid_argument(format!("serialize dataset manifest: {error}")))?;
    std::fs::write(output_root.join("dataset.json"), contents)
        .map_err(|error| invalid_argument(format!("write dataset manifest: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int64Array, StringArray};
    use arrow::record_batch::RecordBatch;
    use arrow_ipc::writer::StreamWriter;
    use rivet_data::dataset::DatasetManifest;
    use rivet_data::dataset::source::Dataset;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "rivet-hf-convert-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_arrow(path: &Path, start: i64) {
        let fields = vec![
            Arc::new(Field::new("bytes", DataType::Binary, true)),
            Arc::new(Field::new("path", DataType::Utf8, true)),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("img", DataType::Struct(fields.clone().into()), true),
            Field::new("label", DataType::Int64, false),
        ]));
        let bytes: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(&[start as u8][..]),
            Some(&[(start + 1) as u8][..]),
        ]));
        let paths: ArrayRef = Arc::new(StringArray::from(vec![
            Some(format!("{start}.png")),
            Some(format!("{}.png", start + 1)),
        ]));
        let image = StructArray::new(fields.into(), vec![bytes, paths], None);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(image),
                Arc::new(Int64Array::from(vec![start, start + 1])),
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = StreamWriter::try_new(file, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn converts_all_shards_and_writes_manifest() {
        let root = temp_root();
        let input_a = root.join("train-00000-of-00002.arrow");
        let input_b = root.join("train-00001-of-00002.arrow");
        let output = root.join("native");
        fs::create_dir_all(&root).unwrap();
        write_arrow(&input_a, 0);
        write_arrow(&input_b, 2);

        let report = convert_huggingface_dataset(
            [HuggingFaceSplit::new("train", vec![input_a, input_b])],
            &output,
            &HuggingFaceLanceOptions {
                keep_path: true,
                max_bytes_per_file: 1,
                max_rows_per_file: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.rows_by_split["train"], 4);

        let manifest = DatasetManifest::read(&output).unwrap();
        assert_eq!(manifest.split("train").unwrap().num_rows, Some(4));
        let dataset =
            crate::datasets::lance::LanceImageDataset::open_default(output.join("train.lance"))
                .unwrap();
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.get(3).unwrap().label, 3);
        let data_file_count = fs::read_dir(output.join("train.lance/data"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.path().extension().and_then(|value| value.to_str()) == Some("lance")
            })
            .count();
        assert!(data_file_count > 1, "expected multiple Lance data files");
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn discovers_train_and_test_shards() {
        let root = temp_root();
        fs::create_dir_all(root.join("data")).unwrap();
        fs::write(root.join("data/train-00000-of-00001.arrow"), b"").unwrap();
        fs::write(root.join("data/cifar10-test.arrow"), b"").unwrap();
        let splits = discover_huggingface_splits(&root).unwrap();
        assert_eq!(
            splits
                .iter()
                .map(|split| split.name.as_str())
                .collect::<Vec<_>>(),
            ["test", "train"]
        );
        fs::remove_dir_all(root).ok();
    }
}
