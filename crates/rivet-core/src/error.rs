use crate::dtype::DType;

/// Errors returned by the tensor foundation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("shape element count overflowed usize")]
    ShapeElementCountOverflow,

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

    #[cfg(feature = "cuda")]
    #[error("CUDA initialization failed: {0}")]
    CudaInitializationFailed(String),

    #[cfg(feature = "cuda")]
    #[error("CUDA synchronization failed: {0}")]
    CudaSynchronizationFailed(String),

    #[cfg(feature = "cuda")]
    #[error("CUDA storage backend is not available for {op}")]
    CudaStorageUnavailable { op: &'static str },

    #[cfg(feature = "cuda")]
    #[error("unsupported CUDA operation: {op}")]
    UnsupportedCudaOp { op: &'static str },

    #[cfg(feature = "cuda")]
    #[error("CUDA operation {op} failed: {message}")]
    CudaOperationFailed { op: &'static str, message: String },

    #[error("storage access out of bounds")]
    StorageOutOfBounds,

    #[error("aligned allocation failed: {bytes} bytes with {alignment}-byte alignment")]
    AllocationFailed { bytes: usize, alignment: usize },

    #[error("aligned buffers do not support zero-sized element types")]
    UnsupportedZeroSizedType,

    #[error("aligned buffer is not fully initialized: initialized {initialized} of {len}")]
    UninitializedStorage { initialized: usize, len: usize },

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

    #[error("{op} received negative index {value}")]
    NegativeIndex { op: &'static str, value: i64 },

    #[error("{op} index {index} is out of bounds for dimension size {size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error("{op} requires a non-empty tensor")]
    EmptyTensorForOp { op: &'static str },

    #[error("{op} cannot use tensors that share storage")]
    StorageAliasConflict { op: &'static str },

    #[error("invalid unfold: dim={dim}, size={size}, step={step}, dim_size={dim_size}")]
    InvalidUnfold {
        dim: usize,
        size: usize,
        step: usize,
        dim_size: usize,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
