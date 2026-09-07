use crate::dataset::Dataset;
use crate::errors::{RivetError, RivetResult, invalid_argument};
use crate::sample::EncodedImageSample;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];

#[derive(Clone)]
pub struct ImageFolderSample {
    pub path: PathBuf,
    pub label: i64,
}

pub struct ImageFolderDatasetCore {
    root: PathBuf,
    samples: Vec<ImageFolderSample>,
    classes: Vec<String>,
    class_to_idx: HashMap<String, i64>,
}

impl ImageFolderDatasetCore {
    pub fn new(root: PathBuf) -> RivetResult<Self> {
        if !root.is_dir() {
            return Err(invalid_argument(format!(
                "image folder root must be a directory: {}",
                root.display()
            )));
        }

        let mut classes = direct_child_dirs(&root)?;
        classes.sort();

        let class_to_idx = classes
            .iter()
            .enumerate()
            .map(|(index, class)| (class.clone(), index as i64))
            .collect::<HashMap<_, _>>();

        let mut samples = Vec::new();
        for class in &classes {
            let label = class_to_idx[class];
            let class_dir = root.join(class);
            let mut paths = image_paths_under(&class_dir)?;
            paths.sort();
            samples.extend(
                paths
                    .into_iter()
                    .map(|path| ImageFolderSample { path, label }),
            );
        }

        Ok(Self {
            root,
            samples,
            classes,
            class_to_idx,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    pub fn class_to_idx(&self) -> &HashMap<String, i64> {
        &self.class_to_idx
    }

    pub fn samples(&self) -> &[ImageFolderSample] {
        &self.samples
    }
}

impl Dataset for ImageFolderDatasetCore {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> RivetResult<Self::Item> {
        let sample = self.samples.get(index).ok_or(RivetError::IndexOutOfRange {
            index,
            len: self.samples.len(),
        })?;

        Ok(EncodedImageSample {
            image: std::fs::read(&sample.path)?,
            label: sample.label,
        })
    }
}

fn direct_child_dirs(root: &Path) -> RivetResult<Vec<String>> {
    let mut classes = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let name = entry.file_name().into_string().map_err(|name| {
                invalid_argument(format!(
                    "class directory name is not valid UTF-8: {:?}",
                    name
                ))
            })?;
            classes.push(name);
        }
    }
    Ok(classes)
}

fn image_paths_under(root: &Path) -> RivetResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in WalkDir::new(root) {
        let entry = entry.map_err(invalid_argument)?;
        if entry.file_type().is_file() && is_image_path(entry.path()) {
            paths.push(entry.path().to_path_buf());
        }
    }
    Ok(paths)
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&extension.as_str())
        })
        .unwrap_or(false)
}
