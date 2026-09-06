use pyo3::prelude::*;

mod batch;
mod dataset;
mod errors;
mod image;
mod pipeline;
mod python;
mod runtime;
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
