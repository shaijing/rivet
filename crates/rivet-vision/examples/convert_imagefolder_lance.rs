//! Convert an ImageFolder tree into a Rivet Lance dataset.
//!
//! Expected input layout: `<input>/<split>/<class>/*.{jpg,jpeg,png,bmp,webp}`.
//! Usage:
//!   cargo run -j 12 -p rivet-vision --features lance \
//!     --example convert_imagefolder_lance -- <imagefolder-root> [output-root] [--overwrite]
//!
//! Example:
//!   cargo run -j 12 -p rivet-vision --features lance \
//!     --example convert_imagefolder_lance -- \
//!     /home/lingyu/.data/custom/nuimages_classification \
//!     /home/lingyu/.data/custom/nuimages_classification.lance

use arrow::array::{BinaryArray, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatchIterator;
use rivet_data::dataset::Dataset;
use rivet_vision::datasets::ImageFolderDatasetCore;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const BATCH_SIZE: usize = 1024;

fn usage() -> &'static str {
    "usage: convert_imagefolder_lance <imagefolder-root> [output-root] [--overwrite]"
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input_root = PathBuf::from(args.next().ok_or_else(|| usage().to_owned())?);
    let mut output_root = None;
    let mut overwrite = false;
    for arg in args {
        if arg == "--overwrite" {
            overwrite = true;
        } else if !arg.starts_with("--") && output_root.is_none() {
            output_root = Some(PathBuf::from(arg));
        } else {
            return Err(format!("unknown argument {arg}; {}", usage()).into());
        }
    }
    if !input_root.is_dir() {
        return Err(format!("input root is not a directory: {}", input_root.display()).into());
    }
    let output_root = output_root.unwrap_or_else(|| {
        let mut output = input_root.as_os_str().to_owned();
        output.push(".lance");
        PathBuf::from(output)
    });
    if output_root == input_root || output_root.starts_with(&input_root) {
        return Err("output root must be outside the ImageFolder input tree".into());
    }
    if output_root.exists() {
        if !overwrite {
            return Err(format!(
                "output already exists: {}; pass --overwrite to replace it",
                output_root.display()
            )
            .into());
        }
        if output_root.is_dir() {
            std::fs::remove_dir_all(&output_root)?;
        } else {
            std::fs::remove_file(&output_root)?;
        }
    }
    std::fs::create_dir_all(&output_root)?;

    let mut rows_by_split = BTreeMap::new();
    let mut class_names: Option<Vec<String>> = None;
    let mut split_count = 0usize;
    for entry in std::fs::read_dir(&input_root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let split = entry
            .file_name()
            .into_string()
            .map_err(|name| format!("split name is not UTF-8: {name:?}"))?;
        validate_split(&split)?;
        let dataset = Arc::new(ImageFolderDatasetCore::new(entry.path())?);
        if let Some(expected) = &class_names {
            if expected != dataset.classes() {
                return Err(format!(
                    "class directories differ in split {split}: expected {expected:?}, got {:?}",
                    dataset.classes()
                )
                .into());
            }
        } else {
            class_names = Some(dataset.classes().to_vec());
        }
        if dataset.is_empty() {
            eprintln!("skip empty split: {split}");
            continue;
        }
        let rows = write_split(Arc::clone(&dataset), &output_root.join(format!("{split}.lance")))?;
        println!("{split}: {rows} images");
        rows_by_split.insert(split, rows);
        split_count += 1;
    }
    if split_count == 0 {
        return Err("no non-empty split directories found".into());
    }
    write_manifest(
        &output_root,
        &rows_by_split,
        class_names.as_deref().unwrap_or_default(),
    )?;
    println!("wrote Rivet dataset: {}", output_root.display());
    Ok(())
}

fn validate_split(split: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(split);
    if split.is_empty()
        || path.is_absolute()
        || path.components().count() != 1
        || path.file_name().and_then(|name| name.to_str()) != Some(split)
    {
        return Err(format!("invalid split directory name: {split:?}").into());
    }
    Ok(())
}

fn write_split(
    dataset: Arc<ImageFolderDatasetCore>,
    output_path: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("image", DataType::Binary, false),
        Field::new("label", DataType::Int32, false),
    ]));
    let row_count = dataset.len();
    let batch_schema = Arc::clone(&schema);
    let batches = (0..row_count).step_by(BATCH_SIZE).map(move |start| {
        let end = (start + BATCH_SIZE).min(row_count);
        let indices = (start..end).collect::<Vec<_>>();
        let samples = dataset
            .get_many(&indices)
            .map_err(|error| arrow::error::ArrowError::ExternalError(Box::new(error)))?;
        let images = samples
            .iter()
            .map(|sample| Some(sample.image.as_slice()))
            .collect::<Vec<_>>();
        let labels = samples
            .iter()
            .map(|sample| sample.label as i32)
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![Arc::new(BinaryArray::from(images)), Arc::new(Int32Array::from(labels))],
        )?;
        Ok(batch)
    });
    let reader = RecordBatchIterator::new(batches, schema);
    let uri = output_path
        .to_str()
        .ok_or_else(|| format!("output path is not UTF-8: {}", output_path.display()))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    runtime.block_on(lance::Dataset::write(reader, uri, None))?;
    Ok(row_count)
}

fn write_manifest(
    output_root: &Path,
    rows_by_split: &BTreeMap<String, usize>,
    classes: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let splits = rows_by_split
        .iter()
        .map(|(name, rows)| {
            (
                name.clone(),
                serde_json::json!({"uri": format!("{name}.lance"), "num_rows": rows}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let dataset_name = output_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("imagefolder");
    let manifest = serde_json::json!({
        "format_version": 2,
        "dataset": {"name": dataset_name, "modality": "image"},
        "features": {
            "image": {"type": "image", "column": "image", "representation": "encoded"},
            "label": {"type": "class_label", "column": "label", "dtype": "int32", "names": classes}
        },
        "splits": splits,
        "created_by": {
            "tool": "rivet",
            "version": env!("CARGO_PKG_VERSION"),
            "source_format": "imagefolder"
        }
    });
    std::fs::write(
        output_root.join("dataset.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
