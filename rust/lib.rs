use pyo3::prelude::*;

mod arrow_dataset;
mod batch;
mod dataloader;
mod dataset;
mod decoder;
mod errors;
mod pipeline;
mod python;
mod sample;
mod sampler;

#[pymodule]
fn _rivet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyArrowDataset>()?;
    m.add_class::<python::PyDataLoader>()?;
    m.add_class::<python::PyImagePipeline>()?;
    m.add_function(wrap_pyfunction!(python::read_image_batch, m)?)?;
    Ok(())
}
