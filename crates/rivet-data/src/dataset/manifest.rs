//! Dataset-level semantic metadata stored in `dataset.json`.
//!
//! The manifest describes dataset identity, semantic feature-to-column
//! bindings, and split locations. Arrow/Lance remains the source of truth for
//! physical schemas and storage layout.

use crate::errors::{DataResult, invalid_argument};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

pub const MANIFEST_FILE_NAME: &str = "dataset.json";
pub const LEGACY_MANIFEST_FILE_NAME: &str = "dataset.rivet.json";
const CURRENT_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize)]
pub struct DatasetManifest {
    pub format_version: u32,
    pub dataset: DatasetMetadata,
    pub features: BTreeMap<String, Feature>,
    pub splits: BTreeMap<String, SplitManifest>,
    pub created_by: Option<CreatedBy>,
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Feature {
    Image(ImageFeature),
    ClassLabel(ClassLabelFeature),
    Scalar(ScalarFeature),
    Binary(ColumnFeature),
    Text(ColumnFeature),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFeature {
    pub column: String,
    pub representation: ImageRepresentation,
    #[serde(default)]
    pub encoding: Option<ImageEncoding>,
    #[serde(default)]
    pub layout: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub channels: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageRepresentation {
    Encoded,
    Decoded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEncoding {
    Png,
    Jpeg,
    Webp,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassLabelFeature {
    pub column: String,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub num_classes: Option<u32>,
    #[serde(default)]
    pub names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarFeature {
    pub column: String,
    #[serde(default)]
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnFeature {
    pub column: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitManifest {
    pub uri: String,
    pub num_rows: Option<usize>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub checksum: Option<Checksum>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checksum {
    pub algorithm: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatedBy {
    pub tool: String,
    pub version: String,
    #[serde(default)]
    pub source_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ManifestV2 {
    format_version: u32,
    dataset: DatasetMetadata,
    features: BTreeMap<String, Feature>,
    splits: BTreeMap<String, SplitManifest>,
    #[serde(default)]
    created_by: Option<CreatedBy>,
    #[serde(default)]
    extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ManifestV1 {
    format_version: u32,
    #[serde(default)]
    name: Option<String>,
    splits: BTreeMap<String, SplitManifestV1>,
}

#[derive(Debug, Deserialize)]
struct SplitManifestV1 {
    path: PathBuf,
    #[serde(default)]
    num_rows: Option<usize>,
}

impl DatasetManifest {
    /// Read the canonical `dataset.json`, falling back to the v1 filename.
    pub fn read(root: &Path) -> DataResult<Self> {
        let path = manifest_path(root).ok_or_else(|| {
            invalid_argument(format!(
                "dataset root '{}' does not contain {} or {}",
                root.display(),
                MANIFEST_FILE_NAME,
                LEGACY_MANIFEST_FILE_NAME
            ))
        })?;
        Self::load(path)
    }

    /// Parse one manifest file without opening its storage backend.
    pub fn load(path: impl AsRef<Path>) -> DataResult<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&contents).map_err(|error| {
            invalid_argument(format!(
                "invalid dataset manifest {}: {error}",
                path.display()
            ))
        })?;
        let version = value
            .get("format_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "dataset manifest {} is missing an integer format_version",
                    path.display()
                ))
            })?;
        let version = u32::try_from(version).map_err(|_| {
            invalid_argument(format!(
                "dataset manifest {} has an out-of-range format_version",
                path.display()
            ))
        })?;

        let manifest = match version {
            1 => {
                let legacy: ManifestV1 = serde_json::from_value(value).map_err(|error| {
                    invalid_argument(format!(
                        "invalid v1 dataset manifest {}: {error}",
                        path.display()
                    ))
                })?;
                DatasetManifest {
                    format_version: legacy.format_version,
                    dataset: DatasetMetadata {
                        name: legacy.name.unwrap_or_else(|| "legacy-dataset".to_owned()),
                        version: None,
                        modality: None,
                        task: None,
                    },
                    features: BTreeMap::new(),
                    splits: legacy
                        .splits
                        .into_iter()
                        .map(|(name, split)| {
                            (
                                name,
                                SplitManifest {
                                    uri: split.path.to_string_lossy().into_owned(),
                                    num_rows: split.num_rows,
                                    size_bytes: None,
                                    checksum: None,
                                },
                            )
                        })
                        .collect(),
                    created_by: None,
                    extensions: BTreeMap::new(),
                }
            }
            CURRENT_FORMAT_VERSION => {
                let current: ManifestV2 = serde_json::from_value(value).map_err(|error| {
                    invalid_argument(format!(
                        "invalid v2 dataset manifest {}: {error}",
                        path.display()
                    ))
                })?;
                DatasetManifest {
                    format_version: current.format_version,
                    dataset: current.dataset,
                    features: current.features,
                    splits: current.splits,
                    created_by: current.created_by,
                    extensions: current.extensions,
                }
            }
            other => {
                return Err(invalid_argument(format!(
                    "unsupported dataset manifest format_version {other}; supported versions are 1 and {CURRENT_FORMAT_VERSION}"
                )));
            }
        };
        manifest.validate_structure(path)?;
        Ok(manifest)
    }

    pub fn split(&self, name: &str) -> DataResult<&SplitManifest> {
        self.splits
            .get(name)
            .ok_or_else(|| invalid_argument(format!("dataset manifest has no split '{name}'")))
    }

    pub fn feature(&self, name: &str) -> DataResult<&Feature> {
        self.features
            .get(name)
            .ok_or_else(|| invalid_argument(format!("dataset manifest has no feature '{name}'")))
    }

    pub fn split_path(
        &self,
        root: &Path,
        name: &str,
        split: &SplitManifest,
    ) -> DataResult<PathBuf> {
        let path = resolve_local_uri(root, &split.uri).map_err(|message| {
            invalid_argument(format!(
                "manifest split '{name}' has an invalid uri {:?}: {message}",
                split.uri
            ))
        })?;
        if !path.exists() {
            return Err(invalid_argument(format!(
                "dataset manifest references split '{name}' at '{}', but that path does not exist",
                split.uri
            )));
        }
        Ok(path)
    }

    fn validate_structure(&self, path: &Path) -> DataResult<()> {
        if self.splits.is_empty() {
            return Err(invalid_argument(format!(
                "dataset manifest {} does not define any splits",
                path.display()
            )));
        }
        if self.format_version == CURRENT_FORMAT_VERSION {
            if self.dataset.name.trim().is_empty() {
                return Err(invalid_argument("v2 dataset.name must not be empty"));
            }
            if self.features.is_empty() {
                return Err(invalid_argument(
                    "v2 dataset manifest features must not be empty",
                ));
            }
            for (name, split) in &self.splits {
                if split.uri.trim().is_empty() {
                    return Err(invalid_argument(format!(
                        "v2 dataset manifest split '{name}' uri must not be empty"
                    )));
                }
                if split.num_rows.is_none() {
                    return Err(invalid_argument(format!(
                        "v2 dataset manifest split '{name}' must define num_rows"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Prefer the v2 canonical filename but continue to recognize v1 manifests.
pub fn manifest_path(root: &Path) -> Option<PathBuf> {
    [MANIFEST_FILE_NAME, LEGACY_MANIFEST_FILE_NAME]
        .into_iter()
        .map(|name| root.join(name))
        .find(|path| path.is_file())
}

fn resolve_local_uri(root: &Path, uri: &str) -> Result<PathBuf, &'static str> {
    if let Some(path) = uri.strip_prefix("file://") {
        let path = PathBuf::from(path);
        return path
            .is_absolute()
            .then_some(path)
            .ok_or("file URI must be absolute");
    }
    if uri.contains("://") {
        return Err("only local relative paths and file:// URIs are supported");
    }
    let path = PathBuf::from(uri);
    if path.as_os_str().is_empty() {
        return Err("URI must not be empty");
    }
    if path.is_absolute() {
        return Ok(path);
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err("relative URI must not contain '..'");
    }
    Ok(root.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_v1_and_normalizes_its_split_path() {
        let root = std::env::temp_dir().join(format!("rivet-manifest-v1-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(LEGACY_MANIFEST_FILE_NAME),
            r#"{"format_version":1,"name":"fixture","splits":{"train":{"path":"train.lance","num_rows":2}}}"#,
        )
        .unwrap();
        std::fs::create_dir(root.join("train.lance")).unwrap();

        let manifest = DatasetManifest::read(&root).unwrap();
        assert_eq!(manifest.dataset.name, "fixture");
        assert!(manifest.features.is_empty());
        assert_eq!(manifest.split("train").unwrap().uri, "train.lance");
        assert_eq!(
            manifest
                .split_path(&root, "train", manifest.split("train").unwrap())
                .unwrap(),
            root.join("train.lance")
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn reads_v2_with_semantic_features() {
        let root = std::env::temp_dir().join(format!("rivet-manifest-v2-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join(MANIFEST_FILE_NAME),
            r#"{
              "format_version":2,
              "dataset":{"name":"fixture","version":"1.0","modality":"image"},
              "features":{
                "input":{"type":"image","column":"jpeg","representation":"encoded","encoding":"jpeg"},
                "target":{"type":"class_label","column":"target","dtype":"int64","num_classes":2}
              },
              "splits":{"train":{"uri":"train.lance","num_rows":2}},
              "extensions":{"example.vendor":{"enabled":true}}
            }"#,
        )
        .unwrap();

        let manifest = DatasetManifest::read(&root).unwrap();
        assert_eq!(manifest.dataset.version.as_deref(), Some("1.0"));
        assert!(matches!(
            manifest.feature("input").unwrap(),
            Feature::Image(_)
        ));
        assert_eq!(manifest.extensions.len(), 1);
        std::fs::remove_dir_all(root).ok();
    }
}
