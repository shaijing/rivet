use crate::errors::RivetResult;
use crate::sample::EncodedImageSample;
use std::sync::Arc;

mod arrow;
mod image_folder;

pub use arrow::ArrowImageDatasetCore;
pub use image_folder::{ImageFolderDatasetCore, ImageFolderSample};

pub trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> RivetResult<Self::Item>;
}

pub trait ImageDataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> RivetResult<EncodedImageSample>;
}

impl<T> ImageDataset for T
where
    T: Dataset<Item = EncodedImageSample> + Send + Sync,
{
    fn len(&self) -> usize {
        Dataset::len(self)
    }

    fn get(&self, index: usize) -> RivetResult<EncodedImageSample> {
        Dataset::get(self, index)
    }
}

#[derive(Clone)]
pub struct ImageSource {
    inner: Arc<dyn ImageDataset>,
}

impl ImageSource {
    pub fn new<T>(dataset: Arc<T>) -> Self
    where
        T: ImageDataset + 'static,
    {
        Self { inner: dataset }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn get(&self, index: usize) -> RivetResult<EncodedImageSample> {
        self.inner.get(index)
    }
}
