use crate::dtype::DType;

/// Errors returned by the tensor foundation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("shape mismatch: lhs={lhs:?}, rhs={rhs:?}")]
    ShapeMismatchBinary { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("dtype mismatch: lhs={lhs:?}, rhs={rhs:?}")]
    DTypeMismatch { lhs: DType, rhs: DType },

    #[error("invalid dimension {dim} for rank {rank}")]
    InvalidDim { dim: usize, rank: usize },

    #[error("invalid narrow: dim={dim}, start={start}, len={len}, dim_size={dim_size}")]
    InvalidNarrow {
        dim: usize,
        start: usize,
        len: usize,
        dim_size: usize,
    },

    #[error("invalid permutation {0:?}")]
    InvalidPermutation(Vec<usize>),

    #[error("cannot reshape {from:?} into {to:?}")]
    InvalidReshape { from: Vec<usize>, to: Vec<usize> },

    #[error("unexpected dtype: expected {expected:?}, got {actual:?}")]
    UnexpectedDType { expected: DType, actual: DType },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDType { dtype: DType, op: &'static str },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDTypeForOp { op: &'static str, dtype: DType },

    #[error("division by zero for dtype {dtype:?}")]
    DivisionByZero { dtype: DType },

    #[error("device mismatch")]
    DeviceMismatch,

    #[error("storage access out of bounds")]
    StorageOutOfBounds,

    #[error("invalid layout: shape rank is {rank}, stride rank is {stride_len}")]
    InvalidLayout { rank: usize, stride_len: usize },

    #[error("cannot broadcast shapes {lhs:?} and {rhs:?}")]
    InvalidBroadcast { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("invalid rank: expected {expected}, got {actual}")]
    InvalidRank { expected: usize, actual: usize },

    #[error("invalid concat dimension {dim} for rank {rank}")]
    InvalidConcatDim { dim: usize, rank: usize },

    #[error("chunk count must be greater than zero, got {chunks}")]
    InvalidChunkCount { chunks: usize },

    #[error("tensor list cannot be empty")]
    EmptyTensorList,

    #[error("range step must not be zero")]
    InvalidRangeStep,

    #[error("range progression overflowed or stopped making progress")]
    RangeOverflow,

    #[error("cannot apply {op} to an empty reduction along dimension {dim}")]
    EmptyReduction { op: &'static str, dim: usize },

    #[error(
        "reduction {op} along dimension {dim} requires at least {minimum} elements, got {actual}"
    )]
    InvalidReduction {
        op: &'static str,
        dim: usize,
        minimum: usize,
        actual: usize,
    },

    #[error("matmul shape mismatch: lhs={lhs:?}, rhs={rhs:?}")]
    MatmulShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("unsupported matmul dtype {dtype:?}")]
    UnsupportedMatmulDType { dtype: DType },

    #[error("unsupported matmul layout")]
    UnsupportedMatmulLayout,
}

pub type Result<T> = std::result::Result<T, Error>;
