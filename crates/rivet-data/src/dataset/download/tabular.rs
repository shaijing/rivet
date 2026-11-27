use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::{fs, io};

use glob::{MatchOptions, Pattern};
use serde_json::json;
use walkdir::WalkDir;

use super::{
    DataFiles, DownloadOptions, cache_key, publish_derived, relative_path, write_raw_link,
};
use crate::dataset::prepare::convert_lance_split;
use crate::dataset::{
    DatasetManifest, PreparedFormat, PreparedWriter, convert_files, manifest_path,
};
use crate::errors::{DataResult, invalid_argument};

const SPLITS: [&str; 5] = ["train", "test", "dev", "val", "validation"];

pub(super) fn derive(
    cache: &Path,
    name: &str,
    source_key: &str,
    options: &DownloadOptions,
    format: PreparedFormat,
    download: impl FnOnce() -> DataResult<PathBuf>,
) -> DataResult<PathBuf> {
    let files = match &options.data_files {
        Some(DataFiles::File(path)) => json!(path),
        Some(DataFiles::Splits(splits)) => {
            if splits.is_empty() {
                return Err(invalid_argument(
                    "data_files must declare at least one split",
                ));
            }
            for split in splits.keys() {
                if split.is_empty()
                    || !split
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
                {
                    return Err(invalid_argument(format!("invalid split name: {split:?}")));
                }
            }
            json!(splits)
        }
        None => json!(null),
    };
    if let Some(columns) = &options.json_columns {
        let names: HashSet<_> = columns.iter().collect();
        if columns.is_empty()
            || columns.iter().any(String::is_empty)
            || names.len() != columns.len()
        {
            return Err(invalid_argument(
                "json_columns must contain unique, nonempty column names",
            ));
        }
    }
    let name: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let name: String = name.trim_matches('-').chars().take(80).collect();
    let name = if name.is_empty() { "dataset" } else { &name };
    let key = cache_key(&json!([
        2,
        source_key,
        files,
        options.json_columns,
        format.as_str()
    ]));
    let output = cache.join(name).join(key);
    if output.join("dataset.json").is_file() {
        return Ok(output);
    }

    let raw = download()?;
    let extraction = (raw.is_file() && !is_tabular_file(&raw))
        .then(tempfile::tempdir)
        .transpose()?;
    let payload = if let Some(temp) = &extraction {
        let target = temp.path().join("data");
        if super::archive::unpack(&raw, &target)? {
            target
        } else {
            super::archive::decompress_file(&raw, &target)?.unwrap_or_else(|| raw.clone())
        }
    } else {
        raw.clone()
    };
    if options.data_files.is_none() {
        if let Some(root) = find_lance_root(&payload)? {
            return publish_derived(output, |staging| {
                convert_lance_bundle(&root, staging, name, format)?;
                write_raw_link(staging, cache, &raw)
            });
        }
    }
    let base = if payload.is_dir() {
        &payload
    } else {
        payload.parent().unwrap()
    };
    let groups = select_files(&payload, base, options.data_files.as_ref())?;
    let root = base.canonicalize()?;
    for path in groups.values().flatten() {
        if !path.canonicalize()?.starts_with(&root) {
            return Err(invalid_argument(format!(
                "data file leaves the download directory: {}",
                path.display()
            )));
        }
    }
    publish_derived(output, |staging| {
        convert_files(
            staging,
            name,
            groups,
            options.json_columns.as_deref(),
            !matches!(&options.data_files, Some(DataFiles::Splits(_))),
            format,
        )?;
        write_raw_link(staging, cache, &raw)
    })
}

pub(in crate::dataset) fn is_tabular_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "parquet"
                    | "arrow"
                    | "ipc"
                    | "feather"
                    | "csv"
                    | "tsv"
                    | "jsonl"
                    | "ndjson"
                    | "json"
            )
        })
}

fn select_files(
    payload: &Path,
    base: &Path,
    selection: Option<&DataFiles>,
) -> DataResult<BTreeMap<String, Vec<PathBuf>>> {
    if let Some(DataFiles::File(file)) = selection {
        let path = base.join(relative_path(file)?);
        if !path.is_file() || (payload.is_file() && path != payload) {
            return Err(invalid_argument(format!(
                "downloaded file not found: {file:?}"
            )));
        }
        return Ok(BTreeMap::from([("train".into(), vec![path])]));
    }
    let candidates = if payload.is_file() {
        vec![payload.to_owned()]
    } else {
        let mut paths = Vec::new();
        for entry in WalkDir::new(payload) {
            let entry = entry.map_err(io::Error::from)?;
            if entry.path().is_file() {
                paths.push(entry.into_path());
            }
        }
        paths.sort();
        paths
    };
    let mut groups = BTreeMap::<String, Vec<PathBuf>>::new();
    if let Some(DataFiles::Splits(splits)) = selection {
        let flags = MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: false,
        };
        for (split, patterns) in splits {
            let mut selected = Vec::new();
            let mut seen = HashSet::new();
            for pattern in patterns {
                let normalized = relative_path(pattern)?
                    .components()
                    .filter_map(|part| match part {
                        std::path::Component::Normal(value) => value.to_str(),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("/");
                let pattern = Pattern::new(&normalized).map_err(invalid_argument)?;
                for path in &candidates {
                    let relative = path.strip_prefix(base).unwrap();
                    if pattern.matches_path_with(relative, flags) && seen.insert(path) {
                        selected.push(path.clone());
                    }
                }
            }
            if selected.is_empty() {
                return Err(invalid_argument(format!(
                    "no downloaded files matched split {split:?}"
                )));
            }
            groups.insert(split.clone(), selected);
        }
        return Ok(groups);
    }

    let mut extensions = HashSet::new();
    for path in candidates {
        let relative = path.strip_prefix(base).unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        let extension = path
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase();
        if !is_tabular_file(&path)
            || relative
                .iter()
                .any(|part| part.to_string_lossy().starts_with('.') || part == "__MACOSX")
            || (payload.is_dir()
                && matches!(
                    &*name,
                    "dataset.json"
                        | "dataset.rivet.json"
                        | "dataset_infos.json"
                        | "dataset_info.json"
                ))
            || name.ends_with(".metadata.json")
        {
            continue;
        }
        extensions.insert(extension);
        let stem = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_ascii_lowercase();
        let split = stem
            .split(['-', '_', '.'])
            .find(|part| SPLITS.contains(part))
            .or_else(|| {
                relative
                    .parent()
                    .unwrap()
                    .iter()
                    .rev()
                    .filter_map(|part| part.to_str())
                    .find(|part| SPLITS.contains(part))
            })
            .unwrap_or("train");
        groups.entry(split.into()).or_default().push(path);
    }
    if groups.is_empty() {
        return Err(invalid_argument(
            "no supported data files found in the download",
        ));
    }
    if extensions.len() > 1 {
        return Err(invalid_argument(
            "multiple input formats found; select splits with data_files",
        ));
    }
    Ok(groups)
}

fn find_lance_root(payload: &Path) -> DataResult<Option<PathBuf>> {
    if !payload.is_dir() {
        return Ok(None);
    }
    let mut found = None;
    let mut dirs = WalkDir::new(payload).into_iter();
    while let Some(entry) = dirs.next() {
        let entry = entry.map_err(io::Error::from)?;
        if !entry.file_type().is_dir() {
            continue;
        }
        let path = entry.path();
        if path != payload && entry.file_name().to_string_lossy().starts_with('.') {
            dirs.skip_current_dir();
            continue;
        }
        let has_lance_splits = fs::read_dir(path)?.any(|child| {
            child.is_ok_and(|child| {
                child.path().extension().is_some_and(|ext| ext == "lance")
                    && child.path().join("_versions").is_dir()
            })
        });
        let has_lance_manifest = manifest_path(path).is_some_and(|manifest| {
            manifest
                .file_name()
                .is_some_and(|name| name == "dataset.rivet.json")
                || DatasetManifest::load(manifest)
                    .is_ok_and(|m| m.splits.values().any(|s| s.uri.ends_with(".lance")))
        });
        if path.extension().is_some_and(|ext| ext == "lance")
            || has_lance_splits
            || has_lance_manifest
        {
            if found.replace(path.to_owned()).is_some() {
                return Err(invalid_argument(
                    "download contains multiple Lance dataset roots",
                ));
            }
            dirs.skip_current_dir();
        }
    }
    Ok(found)
}

pub(super) fn convert_lance_bundle(
    root: &Path,
    output: &Path,
    name: &str,
    format: PreparedFormat,
) -> DataResult<()> {
    let manifest = manifest_path(root).map(DatasetManifest::load).transpose()?;
    let paths = if let Some(manifest) = &manifest {
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
            .collect::<DataResult<Vec<_>>>()?
    } else if root.extension().is_some_and(|ext| ext == "lance") {
        vec![("train".to_owned(), root.to_owned(), None)]
    } else {
        let mut paths = Vec::new();
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path.is_dir() && path.extension().is_some_and(|ext| ext == "lance") {
                paths.push((
                    path.file_stem().unwrap().to_string_lossy().into_owned(),
                    path,
                    None,
                ));
            }
        }
        paths
    };

    fs::create_dir_all(output)?;
    let root = root.canonicalize()?;
    let mut splits = BTreeMap::new();
    let writer = PreparedWriter::new(format)?;
    for (split, path, expected) in paths {
        if !path.canonicalize()?.starts_with(&root) {
            return Err(invalid_argument(format!(
                "downloaded dataset path leaves the cache: {}",
                path.display()
            )));
        }
        if !super::archive::safe_component(&split) {
            return Err(invalid_argument(format!("invalid split name: {split:?}")));
        }
        let file = format!("{split}.{}", format.extension());
        let rows = convert_lance_split(&path, &output.join(&file), &writer)?;
        if expected.is_some_and(|expected| rows != expected) {
            return Err(invalid_argument(format!(
                "manifest row count for split {split:?} does not match the source"
            )));
        }
        splits.insert(split, json!({"uri": file, "path": file, "num_rows": rows}));
    }
    let value = if let Some(manifest) = manifest.filter(|m| m.format_version == 2) {
        let mut value = serde_json::to_value(manifest).map_err(io::Error::other)?;
        value["splits"] = json!(splits);
        value
    } else {
        json!({"format_version": 1, "name": name, "splits": splits})
    };
    serde_json::to_writer_pretty(fs::File::create(output.join("dataset.json"))?, &value)
        .map_err(io::Error::other)?;
    Ok(())
}
