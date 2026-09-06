use pyo3::exceptions::{PyIOError, PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rivet_core::errors::RivetError;

pub(crate) fn to_py_err(err: RivetError) -> PyErr {
    match err {
        RivetError::Io(err) => PyIOError::new_err(err.to_string()),
        RivetError::Arrow(err) => PyRuntimeError::new_err(err.to_string()),
        RivetError::InvalidArgument(message)
        | RivetError::InvalidPipeline(message)
        | RivetError::InvalidShape(message) => PyValueError::new_err(message),
        RivetError::Image(err) => PyRuntimeError::new_err(err.to_string()),
        RivetError::IndexOutOfRange { index, len } => {
            PyIndexError::new_err(format!("index {index} is out of range for length {len}"))
        }
    }
}
