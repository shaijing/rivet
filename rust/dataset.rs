use pyo3::prelude::*;

pub(crate) trait Dataset: Send + Sync {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> PyResult<Self::Item>;
}
