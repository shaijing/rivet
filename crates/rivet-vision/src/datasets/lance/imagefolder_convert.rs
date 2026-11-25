use super::convert::{
    LanceConversionReport, check_output, native_schema, publish_bundle, resolve_directory,
    validate_split_name, write_image_manifest,
};
use crate::datasets::ImageFolderDatasetCore;
use crate::errors::{RivetResult, invalid_argument};
use arrow::array::{BinaryArray, Int32Array};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use rivet_data::dataset::Dataset;
use rivet_data::errors::DataError;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

const BATCH_SIZE: usize = 1024;

/// Convert an ImageFolder tree into a Rivet Lance split bundle.
///
/// With `split`, read class directories directly under the input. Otherwise
/// read one ImageFolder per immediate split directory and require the same
/// class ordering across splits. Encoded image bytes are copied as-is.
pub fn convert_imagefolder_dataset(
    input_root: impl AsRef<Path>,
    output_root: impl AsRef<Path>,
    split: Option<&str>,
    overwrite: bool,
) -> RivetResult<LanceConversionReport> {
    let input = input_root
        .as_ref()
        .canonicalize()
        .map_err(DataError::from)?;
    let output = std::path::absolute(output_root.as_ref()).map_err(DataError::from)?;
    let name = output
        .file_name()
        .ok_or_else(|| invalid_argument("output must name a dataset directory"))?;
    let parent = resolve_directory(output.parent().expect("absolute path has a parent"))
        .map_err(DataError::from)?;
    let output = parent.join(name);
    if output.starts_with(&input) || input.starts_with(&output) {
        return Err(invalid_argument(
            "output must be outside the ImageFolder input tree and its ancestors",
        ));
    }
    check_output(&output, overwrite)?;

    let roots = if let Some(split) = split {
        validate_split_name(split)?;
        BTreeMap::from([(split.to_owned(), input)])
    } else {
        let mut roots = BTreeMap::new();
        for entry in fs::read_dir(input).map_err(DataError::from)? {
            let entry = entry.map_err(DataError::from)?;
            if entry.file_type().map_err(DataError::from)?.is_dir() {
                let split = entry.file_name().into_string().map_err(|name| {
                    invalid_argument(format!("split name is not valid UTF-8: {name:?}"))
                })?;
                validate_split_name(&split)?;
                roots.insert(split, entry.path());
            }
        }
        roots
    };
    let mut datasets = BTreeMap::new();
    for (split, root) in roots {
        let dataset = ImageFolderDatasetCore::new(root)?;
        if !dataset.is_empty() {
            datasets.insert(split, dataset);
        }
    }
    let classes = datasets
        .values()
        .next()
        .ok_or_else(|| invalid_argument("no non-empty ImageFolder splits found"))?
        .classes()
        .to_vec();
    for (split, dataset) in &datasets {
        if dataset.classes() != classes {
            return Err(invalid_argument(format!(
                "class directories differ in split {split}: expected {classes:?}, got {:?}",
                dataset.classes()
            )));
        }
    }

    fs::create_dir_all(&parent).map_err(DataError::from)?;
    let temp = tempfile::Builder::new()
        .prefix(".rivet-imagefolder-")
        .tempdir_in(parent)
        .map_err(DataError::from)?;
    let staging = temp.path().join("converted").join(name);
    fs::create_dir_all(&staging).map_err(DataError::from)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(DataError::from)?;
    let schema = native_schema(false);
    let mut rows_by_split = BTreeMap::new();
    for (split, dataset) in datasets {
        let path = staging.join(format!("{split}.lance"));
        let uri = path.to_str().ok_or_else(|| {
            invalid_argument(format!(
                "output path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let rows = dataset.len();
        let batch_schema = Arc::clone(&schema);
        let batches = (0..rows).step_by(BATCH_SIZE).map(move |start| {
            let indices: Vec<_> = (start..(start + BATCH_SIZE).min(rows)).collect();
            let samples = dataset
                .get_many(&indices)
                .map_err(|err| ArrowError::ExternalError(Box::new(err)))?;
            let images =
                BinaryArray::from_iter_values(samples.iter().map(|sample| sample.image.as_slice()));
            let labels = samples
                .iter()
                .map(|sample| i32::try_from(sample.label))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| ArrowError::ExternalError(Box::new(err)))?;
            RecordBatch::try_new(
                batch_schema.clone(),
                vec![Arc::new(images), Arc::new(Int32Array::from(labels))],
            )
        });
        let reader = RecordBatchIterator::new(batches, schema.clone());
        runtime
            .block_on(lance::Dataset::write(reader, uri, None))
            .map_err(DataError::from)?;
        rows_by_split.insert(split, rows);
    }
    write_image_manifest(
        &staging,
        &rows_by_split,
        false,
        "imagefolder",
        Some(&classes),
    )?;

    publish_bundle(temp, &staging, &output, overwrite)?;
    Ok(LanceConversionReport {
        output_root: output,
        rows_by_split,
    })
}
