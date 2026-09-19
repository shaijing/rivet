use std::fmt;

pub type VisionResult<T> = Result<T, VisionError>;
pub type RivetResult<T> = VisionResult<T>;
pub type RivetError = VisionError;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error(transparent)]
    Data(#[from] rivet_data::DataError),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Core(#[from] rivet_core::Error),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("invalid pipeline: {0}")]
    InvalidPipeline(String),
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("worker error: {0}")]
    Worker(String),
}

pub fn invalid_argument(message: impl fmt::Display) -> VisionError {
    VisionError::InvalidArgument(message.to_string())
}

pub fn invalid_pipeline(message: impl fmt::Display) -> VisionError {
    VisionError::InvalidPipeline(message.to_string())
}

pub fn invalid_shape(message: impl fmt::Display) -> VisionError {
    VisionError::InvalidShape(message.to_string())
}
