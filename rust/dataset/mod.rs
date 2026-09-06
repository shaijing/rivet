use pyo3::prelude::*;

mod arrow;

pub(crate) use arrow::ArrowImageDatasetCore;

pub(crate) trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> PyResult<Self::Item>;
}
