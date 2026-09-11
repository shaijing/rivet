//! Dataset-level metadata stored in dataset.rivet.json.

use crate::errors::{RivetResult, invalid_argument};
use serde::Deserialize;
use std::path::{Component, Path, PathBuf};

pub const MANIFEST_FILE_NAME: &str = "dataset.rivet.json";
const CURRENT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
pub struct DatasetManifest {
    pub format_version: u32,
    pub name: Option<String>,
    pub splits: std::collections::BTreeMap<String, SplitManifest>,
}

#[derive(Debug, Deserialize)]
pub struct SplitManifest {
    pub path: PathBuf,
    pub num_rows: Option<usize>,
}

impl DatasetManifest {
    pub fn read(root: &Path) -> RivetResult<Self> {
        let path = root.join(MANIFEST_FILE_NAME);
        let contents = std::fs::read_to_string(&path)?;
        let manifest: Self = serde_json::from_str(&contents).map_err(|error| {
            invalid_argument(format!(
                "invalid dataset manifest {}: {error}",
                path.display()
            ))
        })?;

        if manifest.format_version != CURRENT_FORMAT_VERSION {
            return Err(invalid_argument(format!(
                "unsupported dataset manifest format_version {}; supported version is {}",
                manifest.format_version, CURRENT_FORMAT_VERSION
            )));
        }
        if manifest.splits.is_empty() {
            return Err(invalid_argument(format!(
                "dataset manifest {} does not define any splits",
                path.display()
            )));
        }

        Ok(manifest)
    }

    pub fn split_path(
        &self,
        root: &Path,
        name: &str,
        split: &SplitManifest,
    ) -> RivetResult<PathBuf> {
        if split.path.as_os_str().is_empty()
            || split.path.is_absolute()
            || split
                .path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(invalid_argument(format!(
                "manifest split '{name}' has an unsafe relative path: {}",
                split.path.display()
            )));
        }

        let path = root.join(&split.path);
        if !path.exists() {
            return Err(invalid_argument(format!(
                "dataset manifest references split '{name}' at '{}', but that path does not exist",
                split.path.display()
            )));
        }
        Ok(path)
    }
}
