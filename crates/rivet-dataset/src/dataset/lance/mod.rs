//! Lance-backed datasets.
//!
//! The storage layer exposes a synchronous `take` bridge because Rivet's
//! public loader is synchronous. Image adaptation stays separate from the
//! generic table reader so future modalities can reuse the same projection
//! and runtime code.

mod image;
mod table;

use super::bundle::{DatasetBundle, DatasetLoadResult};
use super::image_source::ImageSource;
use super::manifest::{DatasetManifest, MANIFEST_FILE_NAME};
use crate::dataset::source::Dataset;
use crate::errors::{RivetResult, invalid_argument};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use image::LanceImageDataset;
pub use table::LanceTable;

/// Open a physical Lance dataset or a logical root containing split datasets.
///
/// A path ending in .lance is always treated as one physical dataset. Other
/// directories use dataset.rivet.json when present, then fall back to
/// discovering immediate child directories named <split>.lance.
pub fn load_lance_image_dataset(
    path: impl AsRef<Path>,
    image_column: impl Into<String>,
    label_column: impl Into<String>,
) -> RivetResult<DatasetLoadResult> {
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
        return Err(invalid_argument(format!(
            "dataset root does not exist: {}",
            path.display()
        )));
    }
    if !path.is_dir() {
        return Err(invalid_argument(format!(
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
                return Err(invalid_argument(format!(
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

fn discover_split_paths(root: &Path) -> RivetResult<Vec<(String, PathBuf, Option<usize>)>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() || path.extension() != Some(OsStr::new("lance")) {
            continue;
        }
        let Some(name) = path.file_stem().and_then(OsStr::to_str) else {
            continue;
        };
        paths.push((name.to_owned(), path, None));
    }
    paths.sort_by(|left, right| left.0.cmp(&right.0));

    if paths.is_empty() {
        return Err(invalid_argument(format!(
            "dataset root '{}' contains no .lance split directories and no {}",
            root.display(),
            MANIFEST_FILE_NAME
        )));
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BinaryArray, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatchIterator;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rivet-lance-bundle-{}-{}",
            std::process::id(),
            NEXT_TMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_fixture(path: &Path, rows: i32) {
        std::fs::create_dir_all(path).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("image", DataType::Binary, false),
            Field::new("label", DataType::Int32, false),
        ]));
        let images: BinaryArray = (0..rows)
            .map(|row| Some(format!("image-{row}").into_bytes()))
            .collect();
        let labels = Int32Array::from_iter_values(0..rows);
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(images), Arc::new(labels)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(lance::Dataset::write(reader, path.to_str().unwrap(), None))
            .unwrap();
    }

    #[test]
    fn discovers_immediate_lance_splits_in_sorted_order() {
        let root = temp_root();
        write_fixture(&root.join("train_1pct.lance"), 2);
        write_fixture(&root.join("test.lance"), 3);
        std::fs::create_dir_all(root.join("nested/ignored.lance")).unwrap();

        let loaded = load_lance_image_dataset(&root, "image", "label").unwrap();
        let DatasetLoadResult::Bundle(bundle) = loaded else {
            panic!("expected a split bundle");
        };
        assert_eq!(
            bundle.split_names().collect::<Vec<_>>(),
            ["test", "train_1pct"]
        );
        assert_eq!(bundle.split("test").unwrap().len(), 3);
        assert!(bundle.split("ignored").is_none());

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn manifest_takes_precedence_over_discovery() {
        let root = temp_root();
        write_fixture(&root.join("train.lance"), 2);
        write_fixture(&root.join("validation.lance"), 3);
        write_fixture(&root.join("ignored.lance"), 4);
        std::fs::write(
            root.join(MANIFEST_FILE_NAME),
            r#"{
                "format_version": 1,
                "name": "fixture",
                "splits": {
                    "train": {"path": "train.lance", "num_rows": 2},
                    "validation": {"path": "validation.lance", "num_rows": 3}
                }
            }"#,
        )
        .unwrap();

        let loaded = load_lance_image_dataset(&root, "image", "label").unwrap();
        let DatasetLoadResult::Bundle(bundle) = loaded else {
            panic!("expected a split bundle");
        };
        assert_eq!(
            bundle.split_names().collect::<Vec<_>>(),
            ["train", "validation"]
        );
        assert!(bundle.split("ignored").is_none());

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn direct_lance_path_returns_one_physical_dataset() {
        let root = temp_root();
        let split = root.join("train.lance");
        write_fixture(&split, 2);

        let loaded = load_lance_image_dataset(&split, "image", "label").unwrap();
        let DatasetLoadResult::Single(dataset) = loaded else {
            panic!("expected one physical dataset");
        };
        assert_eq!(dataset.len(), 2);

        std::fs::remove_dir_all(root).ok();
    }
}
