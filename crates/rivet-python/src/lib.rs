use pyo3::prelude::*;

mod dataset;
mod dlpack;
mod dtype;
mod error;
mod loader;
mod numpy;
mod pipeline;
mod structured;

#[pymodule]
fn _rivet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dlpack::PyDLPackTensor>()?;
    m.add_class::<dataset::PyArrowDataset>()?;
    #[cfg(feature = "lance")]
    m.add_class::<dataset::PyLanceDataset>()?;
    #[cfg(feature = "lance")]
    m.add_class::<dataset::PyLanceDatasetDict>()?;
    m.add_class::<dataset::PyImageFolderDataset>()?;
    m.add_class::<loader::PyDataLoader>()?;
    m.add_class::<pipeline::PyTransform>()?;
    m.add_class::<pipeline::PyImagePipeline>()?;
    m.add_class::<structured::PyAnnotatedSample>()?;
    m.add_class::<structured::PyStructuredPipeline>()?;
    m.add_class::<structured::PyStructuredLoader>()?;
    m.add_class::<structured::PyVideoSample>()?;
    m.add_function(wrap_pyfunction!(structured::mix_batch, m)?)?;
    m.add_function(wrap_pyfunction!(structured::temporal_subsample, m)?)?;
    m.add_function(wrap_pyfunction!(loader::read_image_batch, m)?)?;
    #[cfg(feature = "lance")]
    m.add_function(wrap_pyfunction!(dataset::load_lance_split, m)?)?;
    #[cfg(feature = "lance")]
    m.add_function(wrap_pyfunction!(dataset::_convert_lance_files, m)?)?;
    #[cfg(feature = "lance")]
    m.add_function(wrap_pyfunction!(dataset::_convert_imagefolder_lance, m)?)?;
    Ok(())
}
