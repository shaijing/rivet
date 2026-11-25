//! Discover completed downloads and Lance bundles in a Rivet cache.

use super::{DatasetManifest, manifest_path};
use crate::errors::DataResult;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetEntry {
    pub name: String,
    pub path: PathBuf,
    pub kind: &'static str,
    pub provider: Option<&'static str>,
    pub source: Option<String>,
    pub revision: Option<String>,
    pub splits: Vec<String>,
}

#[derive(Deserialize)]
struct ImportIndex {
    datasets: BTreeMap<String, ImportRecord>,
}

#[derive(Clone, Deserialize)]
struct ImportRecord {
    path: PathBuf,
    raw: Option<PathBuf>,
    source: Option<String>,
    revision: Option<String>,
}

/// List completed entries without opening dataset contents or accessing the network.
pub fn list_datasets(root: impl AsRef<Path>) -> DataResult<Vec<DatasetEntry>> {
    let root = match root.as_ref().canonicalize() {
        Ok(root) => root,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    let mut imports = BTreeMap::new();
    for path in children(&root.join("manifests"), false)? {
        if let Ok(contents) = fs::read(path)
            && let Ok(index) = serde_json::from_slice::<ImportIndex>(&contents)
        {
            for (name, record) in index.datasets {
                if let Some(raw) = &record.raw {
                    imports.insert(raw.clone(), (name.clone(), record.clone()));
                }
                imports.insert(record.path.clone(), (name, record));
            }
        }
    }

    let mut results = Vec::new();
    for provider in ["hf", "git", "http", "torch"] {
        for path in children(&root.join(provider), true)? {
            if provider == "torch" {
                for entry in children(&path, true)? {
                    if let Some(item) = download_entry(&entry, provider, Some(&path), &imports)? {
                        results.push(item);
                    }
                }
            } else if let Some(item) = download_entry(&path, provider, None, &imports)? {
                results.push(item);
            }
        }
    }

    for name_root in children(&root.join("derived/lance"), true)? {
        if let Some(item) = lance_entry(&name_root, "derived", None)? {
            results.push(item);
        }
        for path in children(&name_root, true)? {
            if let Some(item) = lance_entry(&path, "derived", Some(&name_root))? {
                results.push(item);
            }
        }
    }
    for path in children(&root, true)?
        .into_iter()
        .chain(children(&root.join("local/lance"), true)?)
    {
        if let Some(item) = lance_entry(&path, "local", None)? {
            results.push(item);
        }
    }
    results.sort_by(|a, b| (&a.name, a.kind, &a.path).cmp(&(&b.name, b.kind, &b.path)));
    Ok(results)
}

fn children(path: &Path, directories: bool) -> io::Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry?;
        let kind = entry.file_type()?;
        if !entry.file_name().to_string_lossy().starts_with('.')
            && ((directories && kind.is_dir()) || (!directories && kind.is_file()))
        {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

fn download_entry(
    entry: &Path,
    provider: &'static str,
    name_root: Option<&Path>,
    imports: &BTreeMap<PathBuf, (String, ImportRecord)>,
) -> DataResult<Option<DatasetEntry>> {
    let marker = entry.join("artifact");
    if !marker.is_file() {
        return Ok(None);
    }
    let artifact = match fs::read_to_string(marker) {
        Ok(value) => value,
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::InvalidData
            ) =>
        {
            return Ok(None);
        }
        Err(err) => return Err(err.into()),
    };
    let relative = Path::new(&artifact);
    if relative.as_os_str().is_empty()
        || !relative
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Ok(None);
    }
    let path = entry.join(relative);
    let resolved = match path.canonicalize() {
        Ok(path) => path,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    if !resolved.starts_with(entry) || !(resolved.is_file() || resolved.is_dir()) {
        return Ok(None);
    }
    let imported = imports.get(&path);
    let name = imported.map(|(name, _)| name.clone()).unwrap_or_else(|| {
        fs::read_to_string(entry.join("dataset-name"))
            .ok()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| {
                display_name(name_root.unwrap_or(if path.is_file() { &path } else { entry }))
            })
    });
    Ok(Some(DatasetEntry {
        name,
        path,
        kind: "download",
        provider: Some(provider),
        source: imported.and_then(|(_, item)| item.source.clone()),
        revision: imported.and_then(|(_, item)| item.revision.clone()),
        splits: Vec::new(),
    }))
}

fn lance_entry(
    path: &Path,
    kind: &'static str,
    name_root: Option<&Path>,
) -> DataResult<Option<DatasetEntry>> {
    let (name, splits) = if manifest_path(path).is_some() {
        let Ok(manifest) = DatasetManifest::read(path) else {
            return Ok(None);
        };
        if manifest.splits.iter().any(|(name, split)| {
            manifest
                .split_path(path, name, split)
                .map_or(true, |path| !path.is_dir())
        }) {
            return Ok(None);
        }
        let name = name_root.map_or(manifest.dataset.name, display_name);
        (name, manifest.splits.into_keys().collect())
    } else if path.extension().is_some_and(|ext| ext == "lance") && path.join("_versions").is_dir()
    {
        (display_name(path), Vec::new())
    } else {
        let splits: Vec<_> = children(path, true)?
            .into_iter()
            .filter(|child| {
                child.extension().is_some_and(|ext| ext == "lance")
                    && child.join("_versions").is_dir()
            })
            .filter_map(|child| {
                child
                    .file_stem()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .collect();
        if splits.is_empty() {
            return Ok(None);
        }
        (display_name(name_root.unwrap_or(path)), splits)
    };

    Ok(Some(DatasetEntry {
        name,
        path: path.to_path_buf(),
        kind,
        provider: None,
        source: None,
        revision: None,
        splits,
    }))
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}
