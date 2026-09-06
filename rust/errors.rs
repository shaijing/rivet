use std::error::Error;
use std::fmt;

pub(crate) type RivetResult<T> = Result<T, RivetError>;

#[derive(Debug)]
pub(crate) enum RivetError {
    Io(std::io::Error),
    Arrow(arrow::error::ArrowError),
    InvalidArgument(String),
    InvalidPipeline(String),
    InvalidShape(String),
    Decode(String),
    OutOfBounds { index: usize, len: usize },
}

impl fmt::Display for RivetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Arrow(err) => write!(f, "{err}"),
            Self::InvalidArgument(message) => write!(f, "{message}"),
            Self::InvalidPipeline(message) => write!(f, "{message}"),
            Self::InvalidShape(message) => write!(f, "{message}"),
            Self::Decode(message) => write!(f, "{message}"),
            Self::OutOfBounds { index, len } => {
                write!(f, "index {index} is out of range for length {len}")
            }
        }
    }
}

impl Error for RivetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Arrow(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RivetError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<arrow::error::ArrowError> for RivetError {
    fn from(err: arrow::error::ArrowError) -> Self {
        Self::Arrow(err)
    }
}

pub(crate) fn invalid_argument(message: impl fmt::Display) -> RivetError {
    RivetError::InvalidArgument(message.to_string())
}

pub(crate) fn invalid_pipeline(message: impl fmt::Display) -> RivetError {
    RivetError::InvalidPipeline(message.to_string())
}

pub(crate) fn invalid_shape(message: impl fmt::Display) -> RivetError {
    RivetError::InvalidShape(message.to_string())
}
