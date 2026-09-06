use crate::errors::RivetResult;

mod arrow;

pub(crate) use arrow::ArrowImageDatasetCore;

pub(crate) trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> RivetResult<Self::Item>;
}
