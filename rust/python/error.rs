use crate::errors::RivetError;
use pyo3::exceptions::{PyIOError, PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

impl From<RivetError> for PyErr {
    fn from(err: RivetError) -> Self {
        match err {
            RivetError::Io(err) => PyIOError::new_err(err.to_string()),
            RivetError::Arrow(err) => PyRuntimeError::new_err(err.to_string()),
            RivetError::InvalidArgument(message)
            | RivetError::InvalidPipeline(message)
            | RivetError::InvalidShape(message) => PyValueError::new_err(message),
            RivetError::Decode(message) => PyRuntimeError::new_err(message),
            RivetError::OutOfBounds { index, len } => {
                PyIndexError::new_err(format!("index {index} is out of range for length {len}"))
            }
        }
    }
}
