use super::dataset::LanceImageDataset;
use crate::errors::{RivetResult, invalid_argument as vision_invalid_argument};
use crate::source::ImageSource;
use rivet_data::dataset::source::Dataset;
use rivet_data::dataset::{DatasetBundle, DatasetLoadResult, DatasetManifest, MANIFEST_FILE_NAME};
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

    let split_paths = if path.join(MANIFEST_FILE_NAME).is_file() {
        manifest_split_paths(path)?
    } else {
        discover_split_paths(path)?
    };

    let mut splits = BTreeMap::new();
    for (name, split_path, expected_rows) in split_paths {
        let dataset = Arc::new(LanceImageDataset::open(
            &split_path,
            image_column.clone(),
            label_column.clone(),
        )?);
        if let Some(expected_rows) = expected_rows {
            if dataset.len() != expected_rows {
                return Err(vision_invalid_argument(format!(
                    "manifest row count for split '{name}' is {expected_rows}, but Lance reports {}",
                    dataset.len()
                )));
            }
        }
        splits.insert(name, ImageSource::from_encoded(dataset));
    }

    Ok(DatasetLoadResult::Bundle(DatasetBundle::from_splits(
        splits,
    )))
}

fn is_lance_path(path: &Path) -> bool {
    path.extension() == Some(OsStr::new("lance"))
}

fn manifest_split_paths(root: &Path) -> RivetResult<Vec<(String, PathBuf, Option<usize>)>> {
    let manifest = DatasetManifest::read(root)?;
    manifest
        .splits
        .iter()
        .map(|(name, split)| {
            Ok((
                name.clone(),
                manifest.split_path(root, name, split)?,
                split.num_rows,
            ))
        })
        .collect()
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
