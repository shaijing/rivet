use std::fmt;
use thiserror::Error;

pub type RivetResult<T> = Result<T, RivetError>;

#[derive(Debug, Error)]
pub enum RivetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("invalid pipeline: {0}")]
    InvalidPipeline(String),

    #[error("invalid shape: {0}")]
    InvalidShape(String),

    #[error("index {index} is out of range for length {len}")]
    IndexOutOfRange { index: usize, len: usize },
}

pub fn invalid_argument(message: impl fmt::Display) -> RivetError {
    RivetError::InvalidArgument(message.to_string())
}

pub fn invalid_pipeline(message: impl fmt::Display) -> RivetError {
    RivetError::InvalidPipeline(message.to_string())
}

pub fn invalid_shape(message: impl fmt::Display) -> RivetError {
    RivetError::InvalidShape(message.to_string())
}
