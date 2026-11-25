use crate::{error::to_py_err, numpy::tensor_to_numpy, structured::tensor_from_numpy};
use pyo3::{prelude::*, types::PyDict};
use rivet_vision::{
    RandomStream,
    spectrogram::{SpectrogramAxis, SpectrogramSample},
};

#[pyclass(name = "_SpectrogramSample")]
pub(crate) struct PySpectrogramSample {
    inner: SpectrogramSample,
}

#[pymethods]
impl PySpectrogramSample {
    #[new]
    fn new(spectrogram: &Bound<'_, PyAny>, label: i64) -> PyResult<Self> {
        let inner = SpectrogramSample {
            spectrogram: tensor_from_numpy(spectrogram)?,
            label,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn mask(&self, frequency: bool, max_length: usize, seed: u64) -> PyResult<Self> {
        let axis = if frequency {
            SpectrogramAxis::Frequency
        } else {
            SpectrogramAxis::Time
        };
        let mut rng = RandomStream::from_seed(seed);
        let inner = self
            .inner
            .clone()
            .mask(axis, max_length, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn time_reverse(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().time_reverse().map_err(to_py_err)?,
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item(
            "spectrogram",
            tensor_to_numpy(py, self.inner.spectrogram.clone())?,
        )?;
        out.set_item("label", self.inner.label)?;
        Ok(out)
    }
}
