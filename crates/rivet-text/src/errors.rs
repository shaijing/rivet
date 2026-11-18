use thiserror::Error;

pub type TextResult<T> = Result<T, TextError>;

#[derive(Debug, Error)]
pub enum TextError {
    #[error("invalid text pipeline: {0}")]
    InvalidArgument(String),
    #[error("tokenizer {operation} failed: {message}")]
    Tokenizer {
        operation: &'static str,
        message: String,
    },
    #[error("text allocation failed: {0}")]
    Allocation(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
    #[error(transparent)]
    Data(#[from] rivet_data::errors::DataError),
    #[error(transparent)]
    Tensor(#[from] rivet_core::Error),
    #[error("text planning failed: {0}")]
    Plan(String),
    #[error("text batch for source indices {indices:?} failed: {source}")]
    Batch {
        indices: Vec<usize>,
        #[source]
        source: Box<TextError>,
    },
}

pub(crate) fn invalid(message: impl Into<String>) -> TextError {
    TextError::InvalidArgument(message.into())
}
pub(crate) fn tokenizer_error(operation: &'static str, error: impl std::fmt::Display) -> TextError {
    TextError::Tokenizer {
        operation,
        message: error.to_string(),
    }
}
