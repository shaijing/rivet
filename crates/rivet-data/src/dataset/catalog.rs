//! Discover completed downloads and prepared bundles in a Rivet cache.

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

/// List completed entries without opening dataset contents or accessing the network.
pub fn list_datasets(root: impl AsRef<Path>) -> DataResult<Vec<DatasetEntry>> {
    let requested = std::path::absolute(root.as_ref())?;
    let downloads = super::download::download_cache_for(&requested);
    let root = match requested.canonicalize() {
        Ok(root) => root,
        Err(err) if err.kind() == io::ErrorKind::NotFound => requested,
        Err(err) => return Err(err.into()),
    };
    let downloads = match downloads.canonicalize() {
        Ok(root) => root,
        Err(err) if err.kind() == io::ErrorKind::NotFound => downloads,
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

    for name_root in children(&root, true)? {
        for path in children(&name_root, true)? {
            let name = path.file_name().unwrap_or_default().as_encoded_bytes();
            if name.len() == 64
                && name.iter().all(u8::is_ascii_hexdigit)
                && let Some(entry) = prepared_entry(&path, &name_root, &downloads)?
            {
                results.push(entry);
            }
        }
    }
    for path in children(&root, true)?
        .into_iter()
        .chain(children(&root.join("local/lance"), true)?)
    {
        if let Some(item) = lance_entry(&path)? {
            results.push(item);
        }
    }
    let mut seen = std::collections::HashSet::new();
    results.retain(|entry| seen.insert(entry.path.clone()));
    results.sort_by(|a, b| (&a.name, a.kind, &a.path).cmp(&(&b.name, b.kind, &b.path)));
    Ok(results)
}

fn prepared_entry(
    path: &Path,
    name_root: &Path,
    downloads: &Path,
) -> DataResult<Option<DatasetEntry>> {
    let Ok(manifest) = DatasetManifest::read(path) else {
        return Ok(None);
    };
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
        name: display_name(name_root),
        path: path.to_owned(),
        kind: "derived",
        provider: None,
        source: None,
        revision: None,
        splits: manifest.splits.into_keys().collect(),
        formats: formats.into_iter().collect(),
        raw_path: read_raw_link(downloads, path),
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

fn lance_entry(path: &Path) -> DataResult<Option<DatasetEntry>> {
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
        let name = manifest.dataset.name;
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
        (display_name(path), splits)
    };

    Ok(Some(DatasetEntry {
        name,
        path: path.to_path_buf(),
        formats: vec!["lance".into()],
        raw_path: None,
        original_path: None,
        kind: "local",
        provider: None,
        source: None,
        revision: None,
        splits,
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
