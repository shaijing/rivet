//! Download remote datasets into the Rivet cache.

mod archive;
mod remote;
pub(super) mod tabular;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use url::Url;

use tabular::PreparationSource;

use super::{DatasetManifest, PreparedFormat, manifest_path};
use crate::errors::{DataResult, invalid_argument};

#[derive(Debug, Clone, Default)]
pub struct DownloadOptions {
    pub revision: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub offline: bool,
    /// Keep the original HTTP response even when it is an archive.
    pub keep_archive: bool,
    pub format: DownloadFormat,
    pub data_files: Option<DataFiles>,
    pub json_columns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DownloadFormat {
    #[default]
    Raw,
    Lance,
    Arrow,
    ArrowStream,
}

impl DownloadFormat {
    pub fn prepared(self) -> Option<PreparedFormat> {
        match self {
            Self::Raw => None,
            Self::Arrow => Some(PreparedFormat::ArrowIpcFile),
            Self::ArrowStream => Some(PreparedFormat::ArrowIpcStream),
            Self::Lance => Some(PreparedFormat::Lance),
        }
    }
}

impl std::str::FromStr for DownloadFormat {
    type Err = crate::errors::DataError;

    fn from_str(value: &str) -> DataResult<Self> {
        if value == "raw" {
            return Ok(Self::Raw);
        }
        Ok(match value.parse::<PreparedFormat>()? {
            PreparedFormat::ArrowIpcFile => Self::Arrow,
            PreparedFormat::ArrowIpcStream => Self::ArrowStream,
            PreparedFormat::Lance => Self::Lance,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataFiles {
    File(String),
    Splits(BTreeMap<String, Vec<String>>),
}

/// Optional integration for datasets whose native layout is produced by torchvision.
pub trait TorchDatasetProvider {
    fn download(&self, location: &str, destination: &Path) -> DataResult<()>;

    /// Write a Lance bundle to the supplied staging path.
    fn convert(&self, location: &str, raw: &Path, destination: &Path) -> DataResult<()>;

    /// Write an Arrow IPC File bundle to the supplied staging path.
    fn convert_arrow(&self, location: &str, raw: &Path, destination: &Path) -> DataResult<()> {
        self.convert_via_lance(location, raw, destination, PreparedFormat::ArrowIpcFile)
    }

    /// Write a bundle in the selected format.
    fn convert_prepared(
        &self,
        location: &str,
        raw: &Path,
        destination: &Path,
        format: PreparedFormat,
    ) -> DataResult<()> {
        match format {
            PreparedFormat::Lance => self.convert(location, raw, destination),
            PreparedFormat::ArrowIpcFile => self.convert_arrow(location, raw, destination),
            PreparedFormat::ArrowIpcStream => {
                self.convert_via_lance(location, raw, destination, format)
            }
        }
    }

    #[doc(hidden)]
    fn convert_via_lance(
        &self,
        location: &str,
        raw: &Path,
        destination: &Path,
        format: PreparedFormat,
    ) -> DataResult<()> {
        PreparedFormat::Lance.ensure_available()?;
        let native = tempfile::tempdir_in(destination.parent().unwrap())?;
        let bundle = native.path().join("bundle");
        self.convert(location, raw, &bundle)?;
        tabular::convert_bundle(&bundle, destination, "torchvision", format, true)
    }
}

pub fn dataset_cache_dir() -> DataResult<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
        .ok_or_else(|| io::Error::other("cannot determine the dataset cache directory"))?;
    Ok(base.join("rivet/datasets"))
}

pub(in crate::dataset) fn download_cache_for(cache: &Path) -> PathBuf {
    if let Ok(default) = dataset_cache_dir()
        && (default == cache || default.canonicalize().is_ok_and(|path| path == cache))
    {
        return default.with_file_name("downloads");
    }
    cache.join("downloads")
}

fn dataset_cache_for_download(raw_root: &Path, override_root: Option<&Path>) -> Option<PathBuf> {
    if let Some(cache) = override_root
        && let Ok(cache) = std::path::absolute(cache)
        && download_cache_for(&cache)
            .canonicalize()
            .is_ok_and(|root| root == raw_root)
    {
        return Some(cache);
    }
    if let Ok(default) = dataset_cache_dir()
        && download_cache_for(&default)
            .canonicalize()
            .is_ok_and(|root| root == raw_root)
    {
        return Some(default);
    }
    if raw_root.file_name().is_some_and(|part| part == "downloads")
        && let Some(parent) = raw_root.parent()
    {
        return Some(parent.to_owned());
    }
    None
}

pub fn is_remote_dataset(source: &str) -> bool {
    ["hf:", "git:", "github:", "torch:", "http://", "https://"]
        .iter()
        .any(|prefix| source.starts_with(prefix))
}

/// Download an HF snapshot, Git checkout, or HTTP artifact.
///
/// Async callers should run this blocking API on a blocking thread.
pub fn download_dataset(source: &str, options: &DownloadOptions) -> DataResult<PathBuf> {
    download_dataset_with_torch(source, options, None)
}

/// As [`download_dataset`], with an optional provider for `torch:` cache misses.
pub fn download_dataset_with_torch(
    source: &str,
    options: &DownloadOptions,
    torch: Option<&dyn TorchDatasetProvider>,
) -> DataResult<PathBuf> {
    download_with_preparation(source, options, torch, PreparationSource::Download)
}

/// Prepare a remote dataset for loading, reusing an existing compatible bundle.
pub fn prepare_remote_dataset(
    source: &str,
    options: &DownloadOptions,
    torch: Option<&dyn TorchDatasetProvider>,
) -> DataResult<PathBuf> {
    download_with_preparation(source, options, torch, PreparationSource::RemoteLoad)
}

fn download_with_preparation(
    source: &str,
    options: &DownloadOptions,
    torch: Option<&dyn TorchDatasetProvider>,
    preparation: PreparationSource,
) -> DataResult<PathBuf> {
    if options.format == DownloadFormat::Raw
        && (options.data_files.is_some() || options.json_columns.is_some())
    {
        return Err(invalid_argument(
            "data_files and json_columns require a prepared format",
        ));
    }
    if let Some(format) = options.format.prepared() {
        format.ensure_available()?;
    }
    if !is_remote_dataset(source) {
        return Err(invalid_argument(
            "dataset source must use hf:, git:, github:, torch:, http://, or https://",
        ));
    }
    let (provider, location) = source.split_once(':').unwrap();
    let revision = options.revision.as_deref();
    if revision == Some("")
        || (revision.is_some() && matches!(provider, "http" | "https" | "torch"))
    {
        return Err(invalid_argument(
            "revision must be a nonempty Git or Hugging Face reference",
        ));
    }
    let cache = match &options.cache_dir {
        Some(path) => std::path::absolute(path)?,
        None => dataset_cache_dir()?,
    };

    if provider == "torch" {
        return download_torch(location, options, &cache, torch);
    }
    let git_url;
    let (provider, location) = match provider {
        "git" | "github" => {
            git_url = remote::git_url(location)?;
            ("git", git_url.as_str())
        }
        "http" | "https" => ("http", source),
        _ => (provider, location),
    };
    let extract = provider == "http" && !options.keep_archive;
    let mut key = cache_key(&json!([location, revision, extract]));
    let mut entry = download_cache_for(&cache).join(provider).join(&key);
    if extract && options.format.prepared().is_some() && !entry.is_dir() {
        let archive_key = cache_key(&json!([location, revision, false]));
        let archive_entry = download_cache_for(&cache).join(provider).join(&archive_key);
        if archive_entry.join("artifact").is_file() {
            key = archive_key;
            entry = archive_entry;
        }
    }
    let name = location
        .split(['?', '#'])
        .next()
        .unwrap_or_default()
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default();
    let name = name.strip_suffix(".git").unwrap_or(name);
    let raw = || {
        download_raw(
            &entry,
            options.offline,
            if provider == "hf" { location } else { name },
            |payload| {
                match provider {
                    "hf" => remote::hugging_face(location, revision, payload)?,
                    "git" => remote::git(location, payload, revision)?,
                    _ => return remote::http(location, payload, extract),
                }
                Ok(payload.to_owned())
            },
        )
    };

    let Some(format) = options.format.prepared() else {
        return raw();
    };
    tabular::derive(
        &cache,
        name,
        &format!("{provider}/{key}"),
        options,
        format,
        preparation,
        raw,
    )
}

/// Prepare a path previously returned by a raw download. Other local paths
/// are left to the caller's local dataset reader.
pub fn prepare_downloaded_path(
    path: &Path,
    options: &DownloadOptions,
    torch: Option<&dyn TorchDatasetProvider>,
) -> DataResult<Option<PathBuf>> {
    let format = options
        .format
        .prepared()
        .unwrap_or(PreparedFormat::ArrowIpcFile);
    format.ensure_available()?;
    let path = match path.canonicalize() {
        Ok(path) => path,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    for entry in path.ancestors() {
        let Some(parent) = entry.parent() else { break };
        let Some(key) = entry.file_name().and_then(|part| part.to_str()) else {
            continue;
        };
        if key.len() != 64 || !key.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        let provider = parent
            .file_name()
            .and_then(|part| part.to_str())
            .unwrap_or_default();
        let is_torch = parent
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|part| part == "torch");
        if !is_torch && !matches!(provider, "hf" | "git" | "http") {
            continue;
        }
        let Ok(artifact) = fs::read_to_string(entry.join("artifact")) else {
            continue;
        };
        if entry.join(relative_path(&artifact)?) != path {
            continue;
        }
        let mut options = options.clone();
        if options.format == DownloadFormat::Raw {
            options.format = DownloadFormat::Arrow;
        }
        let raw_root = if is_torch {
            parent.parent().and_then(Path::parent).unwrap()
        } else {
            parent.parent().unwrap()
        };
        let Some(cache) = dataset_cache_for_download(raw_root, options.cache_dir.as_deref()) else {
            return Ok(None);
        };
        if is_torch {
            let location = fs::read_to_string(entry.join("torch-location"))?;
            options.offline = true;
            return download_torch(&location, &options, &cache, torch).map(Some);
        }
        let name = fs::read_to_string(entry.join("dataset-name")).unwrap_or_else(|_| {
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
        return tabular::derive(
            &cache,
            name.rsplit('/').next().unwrap_or(&name),
            &format!("{provider}/{key}"),
            &options,
            format,
            PreparationSource::RemoteLoad,
            || Ok(path.clone()),
        )
        .map(Some);
    }
    Ok(None)
}

/// Reuse matching local storage or derive a cached bundle in the requested format.
pub fn prepare_local_dataset(path: &Path, options: &DownloadOptions) -> DataResult<PathBuf> {
    let format = options
        .format
        .prepared()
        .ok_or_else(|| invalid_argument("a prepared format is required to load a dataset"))?;
    format.ensure_available()?;
    let path = path.canonicalize()?;
    let manifest = manifest_path(&path)
        .map(DatasetManifest::load)
        .transpose()?;
    let split_paths = manifest
        .as_ref()
        .map(|manifest| {
            manifest
                .splits
                .iter()
                .map(|(name, split)| manifest.split_path(&path, name, split))
                .collect::<DataResult<Vec<_>>>()
        })
        .transpose()?;
    if options.data_files.is_none() && options.json_columns.is_none() {
        if let Some(splits) = &split_paths {
            let mut matches = true;
            for split in splits {
                matches &= tabular::matches_format(split, format)?;
            }
            if matches {
                return Ok(path);
            }
        } else if tabular::matches_format(&path, format)?
            || (format == PreparedFormat::Lance && tabular::is_lance_split_root(&path)?)
        {
            return Ok(path);
        }
    }

    let cache = match &options.cache_dir {
        Some(path) => std::path::absolute(path)?,
        None => dataset_cache_dir()?,
    };
    fs::create_dir_all(&cache)?;
    let cache = cache.canonicalize()?;
    // Keep derived output outside the source walk, including when cache_dir
    // is the source directory itself.
    let output_cache = cache.join(".local-prepared");
    let mut entries = Vec::new();
    let inputs = if options.data_files.is_none() {
        split_paths.unwrap_or_else(|| vec![path.clone()])
    } else {
        vec![path.clone()]
    };
    if let Some(manifest) = manifest_path(&path) {
        entries.push(json!([manifest, fs::read_to_string(&manifest)?]));
    }
    for input in inputs {
        for entry in walkdir::WalkDir::new(input.canonicalize()?)
            .sort_by_file_name()
            .into_iter()
            .filter_entry(|entry| entry.path() != output_cache)
        {
            let entry = entry.map_err(io::Error::from)?;
            if entry.file_type().is_dir() {
                continue;
            }
            let metadata = fs::metadata(entry.path())?;
            entries.push(json!([entry.path(), metadata.len(), metadata.modified()?]));
        }
    }
    let key = cache_key(&json!([path, entries]));
    let name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("dataset");
    tabular::derive(
        &output_cache,
        name,
        &key,
        options,
        format,
        PreparationSource::Local,
        || Ok(path.clone()),
    )
}

fn download_torch(
    location: &str,
    options: &DownloadOptions,
    cache: &Path,
    torch: Option<&dyn TorchDatasetProvider>,
) -> DataResult<PathBuf> {
    let location_without_prefix = location.strip_prefix("//").unwrap_or(location);
    let (name, query) = location_without_prefix
        .split_once('?')
        .unwrap_or((location_without_prefix, ""));
    fn identifier(value: &str) -> bool {
        value.starts_with(|c: char| c.is_ascii_alphabetic())
            && value
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
    }
    if !identifier(name) {
        return Err(invalid_argument(
            "torch: requires a torchvision dataset class name",
        ));
    }
    let name = name.to_ascii_lowercase();
    if !query.is_empty() && query.split('&').any(|pair| !pair.contains('=')) {
        return Err(invalid_argument("invalid torchvision query"));
    }
    let mut kwargs = BTreeMap::new();
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        if !identifier(&key)
            || matches!(
                &*key,
                "root" | "download" | "transform" | "target_transform" | "transforms" | "loader"
            )
            || kwargs.contains_key(&key)
        {
            return Err(invalid_argument(format!(
                "invalid or repeated torchvision option: {key:?}"
            )));
        }
        kwargs.insert(key, value);
    }
    let key = cache_key(&json!(kwargs));
    let entry = download_cache_for(cache)
        .join("torch")
        .join(&name)
        .join(&key);
    let require_torch =
        || torch.ok_or_else(|| invalid_argument("torch: requires a torchvision dataset provider"));
    let raw = || {
        download_raw(&entry, options.offline, "", |payload| {
            let provider = require_torch()?;
            fs::create_dir(payload)?;
            provider.download(location, payload)?;
            fs::write(payload.parent().unwrap().join("torch-location"), location)?;
            Ok(payload.to_owned())
        })
    };

    let Some(format) = options.format.prepared() else {
        return raw();
    };
    if options.data_files.is_some() || options.json_columns.is_some() {
        return Err(invalid_argument(
            "torch: selects data with torchvision constructor options",
        ));
    }
    if !matches!(
        name.as_str(),
        "cifar10"
            | "cifar100"
            | "mnist"
            | "fashionmnist"
            | "kmnist"
            | "emnist"
            | "usps"
            | "svhn"
            | "stl10"
    ) {
        return Err(invalid_argument(format!(
            "conversion is unavailable for torchvision dataset {name:?}"
        )));
    }
    let prepared_key = cache_key(&json!([1, "torch", key, format.as_str()]));
    let output = cache.join(&name).join(prepared_key);
    if output.join("dataset.json").is_file() {
        return Ok(output);
    }
    let raw = raw()?;
    publish_derived(output, |staging| {
        require_torch()?.convert_prepared(location, &raw, staging, format)?;
        write_raw_link(staging, cache, &raw)
    })
}

fn download_raw(
    entry: &Path,
    offline: bool,
    name: &str,
    fetch: impl FnOnce(&Path) -> DataResult<PathBuf>,
) -> DataResult<PathBuf> {
    let artifact = || -> DataResult<PathBuf> {
        let relative = fs::read_to_string(entry.join("artifact"))?;
        Ok(entry.join(relative_path(&relative)?))
    };
    if entry.is_dir() {
        return artifact();
    }
    if offline {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "dataset source is absent from the Rivet cache",
        )
        .into());
    }
    let parent = entry.parent().unwrap();
    fs::create_dir_all(parent)?;
    let temp = tempfile::Builder::new()
        .prefix(".download-")
        .tempdir_in(parent)?;
    let staging = temp.path().join("entry");
    fs::create_dir(&staging)?;
    let path = fetch(&staging.join("data"))?;
    if !name.is_empty() {
        fs::write(staging.join("dataset-name"), name)?;
    }
    fs::write(
        staging.join("artifact"),
        path.strip_prefix(&staging)
            .unwrap()
            .to_string_lossy()
            .as_bytes(),
    )?;
    publish(&staging, entry, || entry.join("artifact").is_file())?;
    artifact()
}

fn write_raw_link(staging: &Path, cache: &Path, raw: &Path) -> DataResult<()> {
    let relative_to = |base: &Path| {
        raw.strip_prefix(base).map(Path::to_owned).ok().or_else(|| {
            raw.strip_prefix(base.canonicalize().ok()?)
                .map(Path::to_owned)
                .ok()
        })
    };
    let relative = relative_to(&download_cache_for(cache))
        .ok_or_else(|| invalid_argument("original artifact is outside the Rivet download cache"))?;
    fs::write(
        staging.join("download-artifact"),
        relative.to_string_lossy().as_bytes(),
    )?;
    Ok(())
}

fn publish_derived(
    output: PathBuf,
    convert: impl FnOnce(&Path) -> DataResult<()>,
) -> DataResult<PathBuf> {
    let parent = output.parent().unwrap();
    fs::create_dir_all(parent)?;
    let temp = tempfile::Builder::new()
        .prefix(".convert-")
        .tempdir_in(parent)?;
    let staging = temp.path().join("dataset");
    convert(&staging)?;
    if !staging.join("dataset.json").is_file() {
        return Err(io::Error::other("conversion did not produce a dataset manifest").into());
    }
    publish(&staging, &output, || output.join("dataset.json").is_file())?;
    Ok(output)
}

fn publish(staging: &Path, output: &Path, completed: impl FnOnce() -> bool) -> io::Result<()> {
    match fs::rename(staging, output) {
        Err(_) if completed() => Ok(()),
        result => result,
    }
}

fn relative_path(value: &str) -> DataResult<&Path> {
    let path = Path::new(value);
    if path.components().next().is_none()
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_) | Component::CurDir))
        || !path
            .components()
            .any(|part| matches!(part, Component::Normal(_)))
    {
        return Err(invalid_argument(format!(
            "expected a relative dataset path: {value:?}"
        )));
    }
    Ok(path)
}

// Match Python's json.dumps representation so existing raw and derived entries stay reusable.
fn cache_key(value: &Value) -> String {
    fn encode(value: &Value, out: &mut String) {
        use std::fmt::Write;
        match value {
            Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    encode(item, out);
                }
                out.push(']');
            }
            Value::Object(items) => {
                out.push('{');
                for (index, (key, item)) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    encode(&json!(key), out);
                    out.push_str(": ");
                    encode(item, out);
                }
                out.push('}');
            }
            Value::String(value) => {
                for ch in serde_json::to_string(value).unwrap().chars() {
                    match ch {
                        '\u{7f}'..=char::MAX => {
                            for unit in ch.encode_utf16(&mut [0; 2]) {
                                write!(out, "\\u{unit:04x}").unwrap();
                            }
                        }
                        _ => out.push(ch),
                    }
                }
            }
            _ => out.push_str(&value.to_string()),
        }
    }
    let mut encoded = String::new();
    encode(value, &mut encoded);
    hex::encode(Sha256::digest(encoded.as_bytes()))
}

fn parse_url(value: &str) -> DataResult<Url> {
    Url::parse(value).map_err(invalid_argument)
}
