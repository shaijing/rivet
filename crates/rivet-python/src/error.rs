use pyo3::exceptions::{PyIOError, PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rivet_data::DataError;
use rivet_vision::VisionError;

pub(crate) fn to_py_err<E>(err: E) -> PyErr
where
    E: Into<VisionError>,
{
    match err.into() {
        VisionError::Data(DataError::Io(err)) => PyIOError::new_err(err.to_string()),
        VisionError::Data(DataError::Arrow(err)) => PyRuntimeError::new_err(err.to_string()),
        #[cfg(feature = "lance")]
        VisionError::Data(DataError::Lance(err)) => PyRuntimeError::new_err(err.to_string()),
        VisionError::Data(DataError::InvalidArgument(message))
        | VisionError::InvalidArgument(message)
        | VisionError::InvalidPipeline(message)
        | VisionError::InvalidShape(message) => PyValueError::new_err(message),
        VisionError::Data(DataError::IndexOutOfRange { index, len }) => {
            PyIndexError::new_err(format!("index {index} is out of range for length {len}"))
        }
        VisionError::Image(err) => PyRuntimeError::new_err(err.to_string()),
        VisionError::Core(err) => PyRuntimeError::new_err(err.to_string()),
        VisionError::Worker(message) => PyRuntimeError::new_err(message),
    }
}
