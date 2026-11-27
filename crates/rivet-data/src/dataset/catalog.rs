//! Discover completed downloads and prepared bundles in a Rivet cache.

use super::{DatasetManifest, PreparedFormat};
use crate::errors::{DataResult, invalid_argument};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
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
    pub formats: Vec<String>,
    pub raw_path: Option<PathBuf>,
    pub original_path: Option<PathBuf>,
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

fn cache_roots(root: &Path) -> io::Result<(PathBuf, PathBuf)> {
    fn resolve(path: PathBuf) -> io::Result<PathBuf> {
        match path.canonicalize() {
            Ok(resolved) => Ok(resolved),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(path),
            Err(err) => Err(err),
        }
    }

    let root = std::path::absolute(root)?;
    let downloads = super::download::download_cache_for(&root);
    Ok((resolve(root)?, resolve(downloads)?))
}

/// List completed entries without opening dataset contents or accessing the network.
pub fn list_datasets(root: impl AsRef<Path>) -> DataResult<Vec<DatasetEntry>> {
    let (root, downloads) = cache_roots(root.as_ref())?;
    list_entries(&root, &downloads)
}

fn list_entries(root: &Path, downloads: &Path) -> DataResult<Vec<DatasetEntry>> {
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
        for path in children(&downloads.join(provider), true)? {
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

    for name_root in children(root, true)? {
        if let Some(entry) = prepared_entry(&name_root, &name_root, downloads)? {
            results.push(entry);
        }
        for path in children(&name_root, true)? {
            if let Some(entry) = prepared_entry(&path, &name_root, downloads)? {
                results.push(entry);
            }
        }
    }
    results.sort_by(|a, b| (&a.name, a.kind, &a.path).cmp(&(&b.name, b.kind, &b.path)));
    Ok(results)
}

/// Delete cached entries for an exact dataset name and return the number removed.
/// A format limits removal to prepared entries in that format.
pub fn delete_dataset(
    root: impl AsRef<Path>,
    name: &str,
    format: Option<PreparedFormat>,
) -> DataResult<usize> {
    if name.trim().is_empty() {
        return Err(invalid_argument("dataset name must not be empty"));
    }
    let (root, downloads) = cache_roots(root.as_ref())?;
    let entries = list_entries(&root, &downloads)?;
    let originals: BTreeMap<_, _> = entries
        .iter()
        .filter(|entry| entry.kind == "download")
        .map(|entry| (&entry.path, entry.name.as_str()))
        .collect();
    let mut targets = BTreeSet::new();
    for entry in &entries {
        let display_name = entry
            .raw_path
            .as_ref()
            .and_then(|path| originals.get(path).copied())
            .unwrap_or(&entry.name);
        let matches_format = format.is_none_or(|format| {
            entry.kind != "download" && entry.formats.iter().all(|value| value == format.as_str())
        });
        if display_name != name || !matches_format {
            continue;
        }
        let target = if entry.kind == "download" {
            download_entry_root(&downloads, entry)?
        } else {
            if entry.path == root || !entry.path.starts_with(&root) {
                return Err(invalid_argument("dataset is outside the Rivet cache"));
            }
            entry.path.clone()
        };
        match target.canonicalize() {
            Ok(path) if path != target => {
                return Err(invalid_argument(format!(
                    "cache entry resolves through a symbolic link: {}",
                    target.display()
                )));
            }
            Ok(_) => {
                targets.insert(target);
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    if targets.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no matching cached dataset entries for {name:?}"),
        )
        .into());
    }

    let mut removed = 0;
    for target in targets {
        let Some(parent) = target.parent() else {
            continue;
        };
        // Detach the completed entry before removing its contents so readers
        // and retries never discover a partially removed cache entry.
        let trash = match tempfile::Builder::new()
            .prefix(".delete-")
            .tempdir_in(parent)
        {
            Ok(trash) => trash,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        };
        match fs::rename(&target, trash.path().join("entry")) {
            Ok(()) => {
                trash.close()?;
                removed += 1;
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(removed)
}

fn download_entry_root(downloads: &Path, entry: &DatasetEntry) -> DataResult<PathBuf> {
    let relative = entry
        .path
        .strip_prefix(downloads)
        .map_err(invalid_argument)?;
    let provider = entry
        .provider
        .ok_or_else(|| invalid_argument("missing download provider"))?;
    if !relative.starts_with(provider) {
        return Err(invalid_argument(
            "invalid Rivet download provider directory",
        ));
    }
    let depth = if provider == "torch" { 3 } else { 2 };
    let path = downloads.join(relative.components().take(depth).collect::<PathBuf>());
    if path == downloads || !path.join("artifact").is_file() {
        return Err(invalid_argument("invalid Rivet download cache entry"));
    }
    Ok(path)
}

fn prepared_entry(
    path: &Path,
    name_root: &Path,
    downloads: &Path,
) -> DataResult<Option<DatasetEntry>> {
    let Ok(manifest) = DatasetManifest::read(path) else {
        return Ok(None);
    };
    let raw_path = read_raw_link(downloads, path);
    let is_local = raw_path.is_none() && manifest.is_local();
    let key = path.file_name().unwrap_or_default().as_encoded_bytes();
    if !is_local && (path == name_root || key.len() != 64 || !key.iter().all(u8::is_ascii_hexdigit))
    {
        return Ok(None);
    }
    let mut formats = std::collections::BTreeSet::new();
    for (name, split) in &manifest.splits {
        let Ok(split_path) = manifest.split_path(path, name, split) else {
            return Ok(None);
        };
        if split_path.is_file() {
            formats.insert(file_format(&split_path));
        } else if split_path.is_dir() && split_path.extension().is_some_and(|ext| ext == "lance") {
            formats.insert("lance".to_owned());
        } else {
            return Ok(None);
        }
    }
    Ok(Some(DatasetEntry {
        name: if is_local {
            manifest.dataset.name
        } else {
            display_name(name_root)
        },
        path: path.to_owned(),
        kind: if is_local { "local" } else { "derived" },
        provider: None,
        source: None,
        revision: None,
        splits: manifest.splits.into_keys().collect(),
        formats: formats.into_iter().collect(),
        raw_path,
        original_path: None,
    }))
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
    let original = children(&entry.join("original"), false)?;
    let formats = if provider == "torch" {
        vec!["torchvision".into()]
    } else if let Some(archive) = original.first() {
        vec![file_format(archive)]
    } else {
        raw_formats(&path)?
    };
    Ok(Some(DatasetEntry {
        name,
        path,
        formats,
        raw_path: None,
        original_path: original.into_iter().next(),
        kind: "download",
        provider: Some(provider),
        source: imported.and_then(|(_, item)| item.source.clone()),
        revision: imported.and_then(|(_, item)| item.revision.clone()),
        splits: Vec::new(),
    }))
}

fn read_raw_link(downloads: &Path, path: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(path.join("download-artifact")).ok()?;
    let relative = Path::new(&text);
    if relative.as_os_str().is_empty()
        || !relative
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return None;
    }
    Some(downloads.join(relative))
}

fn file_format(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("unknown")
        .to_ascii_lowercase();
    if matches!(ext.as_str(), "gz" | "bz2" | "xz")
        && let Some(inner) = path.file_stem().map(Path::new)
        && let Some(inner_ext) = inner.extension().and_then(|ext| ext.to_str())
        && (super::download::tabular::is_tabular_file(inner)
            || inner_ext.eq_ignore_ascii_case("tar"))
    {
        return format!("{}.{}", inner_ext.to_ascii_lowercase(), ext);
    }
    if matches!(ext.as_str(), "arrow" | "ipc" | "feather") {
        use std::io::Read;
        let mut magic = [0; 6];
        if fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .is_ok()
        {
            return if &magic == b"ARROW1" {
                "arrow-ipc-file"
            } else {
                "arrow-ipc-stream"
            }
            .into();
        }
    }
    ext
}

fn raw_formats(path: &Path) -> io::Result<Vec<String>> {
    if path.is_file() {
        return Ok(vec![file_format(path)]);
    }
    let mut formats = std::collections::BTreeSet::new();
    let mut entries = walkdir::WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| {
            entry.depth() == 0 || !entry.file_name().to_string_lossy().starts_with('.')
        });
    while let Some(entry) = entries.next() {
        let entry = entry.map_err(io::Error::from)?;
        let candidate = entry.path();
        if entry.file_type().is_dir() && candidate.extension().is_some_and(|ext| ext == "lance") {
            formats.insert("lance".to_owned());
            entries.skip_current_dir();
        } else if entry.file_type().is_file()
            && (super::download::tabular::is_tabular_file(candidate)
                || candidate
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "zip"
                                | "tar"
                                | "gz"
                                | "bz2"
                                | "xz"
                                | "txt"
                                | "jpg"
                                | "jpeg"
                                | "png"
                                | "webp"
                                | "wav"
                                | "flac"
                        )
                    }))
            && !matches!(
                entry.file_name().to_str(),
                Some("dataset.json" | "dataset.rivet.json" | "dataset_infos.json")
            )
        {
            formats.insert(file_format(candidate));
        }
    }
    if formats.is_empty() {
        formats.insert("directory".into());
    }
    Ok(formats.into_iter().collect())
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}
