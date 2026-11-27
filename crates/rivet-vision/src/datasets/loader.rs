#[cfg(feature = "lance")]
use super::LanceImageDataset;
use crate::datasets::ArrowImageDataset;
use crate::errors::{RivetResult, invalid_argument as vision_invalid_argument};
use crate::source::ImageSource;
use arrow::datatypes::DataType;
#[cfg(feature = "lance")]
use rivet_data::dataset::MANIFEST_FILE_NAME;
use rivet_data::dataset::source::Dataset;
use rivet_data::dataset::{
    DataFiles, DatasetBundle, DatasetLoadResult, DatasetManifest, DownloadFormat, DownloadOptions,
    Feature, ImageRepresentation, TorchDatasetProvider, download_dataset_with_torch,
    is_remote_dataset, manifest_path, prepare_downloaded_path,
};
#[cfg(feature = "lance")]
use rivet_data::errors::DataResult;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// An explicit local path or a remote dataset source.
#[derive(Debug, Clone, Copy)]
pub enum DatasetLocation<'a> {
    Local(&'a Path),
    Remote(&'a str),
}

impl<'a> From<&'a str> for DatasetLocation<'a> {
    fn from(value: &'a str) -> Self {
        if is_remote_dataset(value) {
            Self::Remote(value)
        } else {
            Self::Local(Path::new(value))
        }
    }
}

impl<'a> From<&'a Path> for DatasetLocation<'a> {
    fn from(value: &'a Path) -> Self {
        Self::Local(value)
    }
}

#[derive(Debug, Clone)]
pub struct LoadDatasetOptions {
    pub split: Option<String>,
    pub revision: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub offline: bool,
    pub image_column: String,
    pub label_column: String,
    pub data_files: Option<DataFiles>,
    pub json_columns: Option<Vec<String>>,
}

impl Default for LoadDatasetOptions {
    fn default() -> Self {
        Self {
            split: None,
            revision: None,
            cache_dir: None,
            offline: false,
            image_column: "image".into(),
            label_column: "label".into(),
            data_files: None,
            json_columns: None,
        }
    }
}

/// Open local image data or prepare and open a remote Arrow IPC bundle.
/// Async callers should run this API on a blocking thread.
pub fn load_dataset<'a>(
    location: impl Into<DatasetLocation<'a>>,
    options: &LoadDatasetOptions,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    load_dataset_with_torch(location, options, None)
}

/// As [`load_dataset`], with a provider for `torch:` cache misses.
pub fn load_dataset_with_torch<'a>(
    location: impl Into<DatasetLocation<'a>>,
    options: &LoadDatasetOptions,
    torch: Option<&dyn TorchDatasetProvider>,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    let download_options = DownloadOptions {
        revision: options.revision.clone(),
        cache_dir: options.cache_dir.clone(),
        offline: options.offline,
        format: DownloadFormat::Arrow,
        data_files: options.data_files.clone(),
        json_columns: options.json_columns.clone(),
        ..DownloadOptions::default()
    };
    let open = |path: &Path, cache_root: Option<&Path>| {
        if options.split.is_some() && is_lance_path(path) && manifest_path(path).is_none() {
            return Err(vision_invalid_argument(
                "split cannot be specified when loading a single .lance dataset path",
            ));
        }
        open_lance_image_dataset(
            path,
            &options.image_column,
            &options.label_column,
            cache_root,
        )
    };
    let loaded = match location.into() {
        DatasetLocation::Local(path) => {
            if options.revision.is_some() {
                return Err(vision_invalid_argument(
                    "revision requires a remote dataset source",
                ));
            }
            if let Some(prepared) = prepare_downloaded_path(path, &download_options, torch)? {
                open_arrow_bundle(&prepared, options)?
            } else if path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| matches!(ext.to_str(), Some("arrow" | "ipc" | "feather")))
            {
                if options.split.is_some() {
                    return Err(vision_invalid_argument(
                        "split cannot be specified when loading a single Arrow IPC path",
                    ));
                }
                let dataset = Arc::new(ArrowImageDataset::new_with_image_alias(
                    vec![path.to_owned()],
                    options.image_column.clone(),
                    options.label_column.clone(),
                    true,
                )?);
                DatasetLoadResult::Single(ImageSource::from_encoded(dataset))
            } else if manifest_path(path).is_some()
                && DatasetManifest::read(path).is_ok_and(|manifest| {
                    manifest
                        .splits
                        .values()
                        .all(|split| split.uri.ends_with(".arrow"))
                })
            {
                open_arrow_bundle(path, options)?
            } else {
                open(path, None)?
            }
        }
        DatasetLocation::Remote(source) => {
            let payload = download_dataset_with_torch(source, &download_options, torch)?;
            open_arrow_bundle(&payload, options)?
        }
    };
    match (loaded, options.split.as_deref()) {
        (DatasetLoadResult::Bundle(bundle), Some(name)) => {
            let dataset =
                bundle
                    .split(name)
                    .ok_or_else(|| crate::VisionError::UnknownDatasetSplit {
                        name: name.to_owned(),
                        available: bundle.split_names().collect::<Vec<_>>().join(", "),
                    })?;
            Ok(DatasetLoadResult::Single(dataset))
        }
        (loaded, _) => Ok(loaded),
    }
}

fn open_arrow_bundle(
    root: &Path,
    options: &LoadDatasetOptions,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    let manifest = DatasetManifest::read(root)?;
    let (image_column, label_column, label_dtype) =
        manifest_columns(&manifest, &options.image_column, &options.label_column)?;
    let cache_root = root.canonicalize().map_err(rivet_data::DataError::from)?;
    let mut splits = BTreeMap::new();
    for (name, split) in &manifest.splits {
        let path = manifest.split_path(root, name, split)?;
        validate_cached_path(&path, Some(&cache_root))?;
        let dataset = Arc::new(ArrowImageDataset::new_with_image_alias(
            vec![path],
            image_column.clone(),
            label_column.clone(),
            manifest.format_version == 1,
        )?);
        validate_label_dtype(dataset.schema(), &label_column, label_dtype.as_deref())?;
        if split.num_rows.is_some_and(|rows| rows != dataset.len()) {
            return Err(vision_invalid_argument(format!(
                "manifest row count for split {name:?} does not match Arrow IPC File"
            )));
        }
        splits.insert(name.clone(), ImageSource::from_encoded(dataset));
    }
    Ok(DatasetLoadResult::Bundle(DatasetBundle::from_splits(
        splits,
    )))
}

fn validate_cached_path(path: &Path, cache_root: Option<&Path>) -> RivetResult<()> {
    if let Some(cache_root) = cache_root
        && !path
            .canonicalize()
            .map_err(rivet_data::DataError::from)?
            .starts_with(cache_root)
    {
        return Err(vision_invalid_argument(format!(
            "downloaded dataset path leaves the cache: {}",
            path.display(),
        )));
    }
    Ok(())
}

/// Open a physical Lance dataset or a logical root containing split datasets.
pub fn load_lance_image_dataset(
    path: impl AsRef<Path>,
    image_column: impl Into<String>,
    label_column: impl Into<String>,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    open_lance_image_dataset(
        path.as_ref(),
        &image_column.into(),
        &label_column.into(),
        None,
    )
}

#[cfg(feature = "lance")]
fn open_lance_image_dataset(
    path: &Path,
    image_column: &str,
    label_column: &str,
    cache_root: Option<&Path>,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    let manifest = manifest_path(path);
    if is_lance_path(path) && manifest.is_none() {
        validate_cached_path(path, cache_root)?;
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

    let split_paths = if let Some(manifest) = manifest {
        validate_cached_path(&manifest, cache_root)?;
        manifest_split_paths(path, &manifest, image_column, label_column)?
    } else {
        discover_split_paths(path)?
            .into_iter()
            .map(|(name, path, num_rows)| ManifestSplitPath {
                name,
                path,
                num_rows,
                image_column: image_column.to_owned(),
                label_column: label_column.to_owned(),
                label_dtype: None,
            })
            .collect()
    };

    let mut splits = BTreeMap::new();
    for split in &split_paths {
        validate_cached_path(&split.path, cache_root)?;
    }
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

#[cfg(not(feature = "lance"))]
fn open_lance_image_dataset(
    _path: &Path,
    _image_column: &str,
    _label_column: &str,
    _cache_root: Option<&Path>,
) -> RivetResult<DatasetLoadResult<ImageSource>> {
    rivet_data::dataset::PreparedFormat::Lance.ensure_available()?;
    unreachable!("Lance feature check")
}

fn is_lance_path(path: &Path) -> bool {
    path.extension() == Some(OsStr::new("lance"))
}

#[cfg(feature = "lance")]
struct ManifestSplitPath {
    name: String,
    path: PathBuf,
    num_rows: Option<usize>,
    image_column: String,
    label_column: String,
    label_dtype: Option<String>,
}

#[cfg(feature = "lance")]
fn manifest_split_paths(
    root: &Path,
    path: &Path,
    fallback_image_column: &str,
    fallback_label_column: &str,
) -> RivetResult<Vec<ManifestSplitPath>> {
    let manifest = DatasetManifest::load(path)?;
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
            "image feature '{name}' uses decoded representation, which the image dataset loader does not yet support"
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
            "class_label feature '{column}' expects {expected}, but the stored schema has {actual:?}"
        )))
    }
}

#[cfg(feature = "lance")]
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
