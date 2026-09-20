use super::dataset::LanceImageDataset;
use crate::errors::{RivetResult, invalid_argument as vision_invalid_argument};
use crate::source::ImageSource;
use arrow::datatypes::DataType;
use rivet_data::dataset::source::Dataset;
use rivet_data::dataset::{
    DatasetBundle, DatasetLoadResult, DatasetManifest, Feature, ImageRepresentation,
    MANIFEST_FILE_NAME, manifest_path,
};
use rivet_data::errors::DataResult;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Open a physical Lance dataset or a logical root containing split datasets.
pub fn load_lance_image_dataset(
    path: impl AsRef<Path>,
    image_column: impl Into<String>,
    label_column: impl Into<String>,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    let path = path.as_ref();
    let image_column = image_column.into();
    let label_column = label_column.into();

    if is_lance_path(path) {
        let dataset = Arc::new(LanceImageDataset::open(path, image_column, label_column)?);
        return Ok(DatasetLoadResult::Single(ImageSource::from_encoded(
            dataset,
        )));
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

    let split_paths = if manifest_path(path).is_some() {
        manifest_split_paths(path, &image_column, &label_column)?
    } else {
        discover_split_paths(path)?
            .into_iter()
            .map(|(name, path, num_rows)| ManifestSplitPath {
                name,
                path,
                num_rows,
                image_column: image_column.clone(),
                label_column: label_column.clone(),
                label_dtype: None,
            })
            .collect()
    };

    let mut splits = BTreeMap::new();
    for split in split_paths {
        let label_column = split.label_column.clone();
        let dataset = Arc::new(LanceImageDataset::open(
            &split.path,
            split.image_column,
            split.label_column,
        )?);
        validate_label_dtype(
            dataset.schema(),
            &label_column,
            split.label_dtype.as_deref(),
        )?;
        if let Some(expected_rows) = split.num_rows {
            if dataset.len() != expected_rows {
                return Err(vision_invalid_argument(format!(
                    "manifest row count for split '{}' is {expected_rows}, but Lance reports {}",
                    split.name,
                    dataset.len()
                )));
            }
        }
        splits.insert(split.name, ImageSource::from_encoded(dataset));
    }

    Ok(DatasetLoadResult::Bundle(DatasetBundle::from_splits(
        splits,
    )))
}

fn is_lance_path(path: &Path) -> bool {
    path.extension() == Some(OsStr::new("lance"))
}

struct ManifestSplitPath {
    name: String,
    path: PathBuf,
    num_rows: Option<usize>,
    image_column: String,
    label_column: String,
    label_dtype: Option<String>,
}

fn manifest_split_paths(
    root: &Path,
    fallback_image_column: &str,
    fallback_label_column: &str,
) -> RivetResult<Vec<ManifestSplitPath>> {
    let manifest = DatasetManifest::read(root)?;
    let (image_column, label_column, label_dtype) =
        manifest_columns(&manifest, fallback_image_column, fallback_label_column)?;
    manifest
        .splits
        .iter()
        .map(|(name, split)| {
            Ok(ManifestSplitPath {
                name: name.clone(),
                path: manifest.split_path(root, name, split)?,
                num_rows: split.num_rows,
                image_column: image_column.clone(),
                label_column: label_column.clone(),
                label_dtype: label_dtype.clone(),
            })
        })
        .collect()
}

fn manifest_columns(
    manifest: &DatasetManifest,
    fallback_image_column: &str,
    fallback_label_column: &str,
) -> RivetResult<(String, String, Option<String>)> {
    if manifest.format_version == 1 {
        return Ok((
            fallback_image_column.to_owned(),
            fallback_label_column.to_owned(),
            None,
        ));
    }

    let images: Vec<_> = manifest
        .features
        .iter()
        .filter_map(|(name, feature)| match feature {
            Feature::Image(image) => Some((name, image)),
            _ => None,
        })
        .collect();
    let [(name, image)] = images.as_slice() else {
        return Err(vision_invalid_argument(
            "v2 image loader requires exactly one feature with type 'image'",
        ));
    };
    if image.representation != ImageRepresentation::Encoded {
        return Err(vision_invalid_argument(format!(
            "image feature '{name}' uses decoded representation, which the Lance image loader does not yet support"
        )));
    }
    let labels: Vec<_> = manifest
        .features
        .iter()
        .filter_map(|(name, feature)| match feature {
            Feature::ClassLabel(label) => Some((name, label)),
            _ => None,
        })
        .collect();
    let [(_, label)] = labels.as_slice() else {
        return Err(vision_invalid_argument(
            "v2 image loader requires exactly one feature with type 'class_label'",
        ));
    };
    Ok((
        image.column.clone(),
        label.column.clone(),
        label.dtype.clone(),
    ))
}

fn validate_label_dtype(
    schema: &arrow::datatypes::Schema,
    column: &str,
    expected: Option<&str>,
) -> RivetResult<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = schema
        .field_with_name(column)
        .map_err(vision_invalid_argument)?
        .data_type();
    let compatible = match expected.trim().to_ascii_lowercase().as_str() {
        "int32" | "i32" => matches!(actual, DataType::Int32),
        "int64" | "i64" => matches!(actual, DataType::Int64),
        other => {
            return Err(vision_invalid_argument(format!(
                "class_label feature '{column}' has unsupported dtype expectation {other:?}"
            )));
        }
    };
    if compatible {
        Ok(())
    } else {
        Err(vision_invalid_argument(format!(
            "class_label feature '{column}' expects {expected}, but Lance schema has {actual:?}"
        )))
    }
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
        return Err(rivet_data::errors::invalid_argument(format!(
            "dataset root '{}' contains no .lance split directories and no {}",
            root.display(),
            MANIFEST_FILE_NAME
        )));
    }
    Ok(paths)
}
