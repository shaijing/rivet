use crate::errors::RivetResult;

mod arrow;

pub use arrow::ArrowImageDatasetCore;

pub trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> RivetResult<Self::Item>;
}
