use pyo3::prelude::*;

mod dataset;
mod error;
mod loader;
mod pipeline;

#[pymodule]
fn _rivet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dataset::PyArrowDataset>()?;
    m.add_class::<dataset::PyImageFolderDataset>()?;
    m.add_class::<loader::PyDataLoader>()?;
    m.add_class::<pipeline::PyImagePipeline>()?;
    m.add_function(wrap_pyfunction!(loader::read_image_batch, m)?)?;
    Ok(())
}
