use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub(crate) fn io_err(err: std::io::Error) -> PyErr {
    PyIOError::new_err(err.to_string())
}

pub(crate) fn runtime_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

pub(crate) fn value_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}
