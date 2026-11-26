//! Download remote datasets into the Rivet cache.

mod archive;
mod remote;
mod tabular;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use url::Url;

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
}

pub fn dataset_cache_dir() -> DataResult<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
        .ok_or_else(|| io::Error::other("cannot determine the dataset cache directory"))?;
    Ok(base.join("rivet/datasets"))
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
    if options.format == DownloadFormat::Raw
        && (options.data_files.is_some() || options.json_columns.is_some())
    {
        return Err(invalid_argument(
            "data_files and json_columns require format='lance'",
        ));
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
    let key = cache_key(&json!([location, revision, extract]));
    let entry = cache.join(provider).join(&key);
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

    if options.format == DownloadFormat::Raw {
        return raw();
    }
    tabular::derive(&cache, name, &format!("{provider}/{key}"), options, raw)
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
    let entry = cache.join("torch").join(&name).join(&key);
    let require_torch =
        || torch.ok_or_else(|| invalid_argument("torch: requires a torchvision dataset provider"));
    let raw = || {
        download_raw(&entry, options.offline, "", |payload| {
            let provider = require_torch()?;
            fs::create_dir(payload)?;
            provider.download(location, payload)?;
            Ok(payload.to_owned())
        })
    };

    if options.format == DownloadFormat::Raw {
        return raw();
    }
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
            "Lance conversion is unavailable for torchvision dataset {name:?}"
        )));
    }
    let output = cache
        .join("derived/lance")
        .join(name)
        .join(format!("torch-v1-{key}"));
    if output.join("dataset.json").is_file() {
        return Ok(output);
    }
    let raw = raw()?;
    publish_derived(output, |staging| {
        require_torch()?.convert(location, &raw, staging)
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
