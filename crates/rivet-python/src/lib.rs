use pyo3::prelude::*;

mod dataset;
mod dlpack;
mod download;
mod dtype;
mod error;
mod loader;
mod numpy;
mod pipeline;
mod spectrogram;
mod structured;
mod text;
mod volume;

#[pymodule]
fn _rivet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    text::register(m)?;
    m.add_class::<dlpack::PyDLPackTensor>()?;
    m.add_class::<dataset::PyArrowDataset>()?;
    m.add_class::<dataset::PyLanceDataset>()?;
    m.add_class::<dataset::PyLanceDatasetDict>()?;
    m.add_class::<dataset::PyImageFolderDataset>()?;
    m.add_class::<loader::PyDataLoader>()?;
    m.add_class::<pipeline::PyTransform>()?;
    m.add_class::<pipeline::PyImagePipeline>()?;
    m.add_class::<structured::PyAnnotatedSample>()?;
    m.add_class::<structured::PyStructuredPipeline>()?;
    m.add_class::<structured::PyStructuredLoader>()?;
    m.add_class::<structured::PyVideoSample>()?;
    m.add_class::<volume::PyVolumeSample>()?;
    m.add_class::<spectrogram::PySpectrogramSample>()?;
    m.add_function(wrap_pyfunction!(structured::mix_batch, m)?)?;
    m.add_function(wrap_pyfunction!(structured::temporal_subsample, m)?)?;
    m.add_function(wrap_pyfunction!(loader::read_image_batch, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::_convert_imagefolder_lance, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::_convert_torch_images, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::_list_datasets, m)?)?;
    m.add_function(wrap_pyfunction!(download::download_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(download::_dataset_cache, m)?)?;
    Ok(())
}
