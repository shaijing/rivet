use std::collections::{BTreeMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

use glob::{MatchOptions, Pattern};
use serde_json::json;
use walkdir::WalkDir;

use super::{DataFiles, DownloadOptions, cache_key, publish_derived, relative_path};
use crate::dataset::lance::convert_files_to_lance;
use crate::errors::{DataResult, invalid_argument};

const SPLITS: [&str; 5] = ["train", "test", "dev", "val", "validation"];

pub(super) fn derive(
    cache: &Path,
    name: &str,
    source_key: &str,
    options: &DownloadOptions,
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
                    return Err(invalid_argument(format!(
                        "invalid Lance split name: {split:?}"
                    )));
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
    let key = cache_key(&json!([1, source_key, files, options.json_columns]));
    let output = cache.join("derived/lance").join(name).join(key);
    if output.join("dataset.json").is_file() {
        return Ok(output);
    }

    let payload = download()?;
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
        convert_files_to_lance(
            staging,
            name,
            groups,
            options.json_columns.as_deref(),
            !matches!(&options.data_files, Some(DataFiles::Splits(_))),
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
        if !matches!(
            extension.as_str(),
            "parquet" | "arrow" | "csv" | "tsv" | "jsonl" | "ndjson" | "json"
        ) || relative
            .iter()
            .any(|part| part.to_string_lossy().starts_with('.') || part == "__MACOSX")
            || (payload.is_dir() && matches!(&*name, "dataset.json" | "dataset.rivet.json"))
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
