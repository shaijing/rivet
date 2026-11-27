use std::fmt;

pub type DataResult<T> = Result<T, DataError>;

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[cfg(feature = "lance")]
    #[error("Lance error: {0}")]
    Lance(#[from] lance::Error),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("index {index} is out of range for length {len}")]
    IndexOutOfRange { index: usize, len: usize },
}

pub fn invalid_argument(message: impl fmt::Display) -> DataError {
    DataError::InvalidArgument(message.to_string())
}
