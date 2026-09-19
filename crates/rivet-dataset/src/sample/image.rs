use arrow_buffer::Buffer;
use rivet_core::Tensor;

use crate::errors::{RivetResult, invalid_argument};

/// Semantic image axis order. The tensor itself remains the source of truth
/// for shape, dtype, strides, and storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Hwc,
    Chw,
}

impl ImageLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hwc => "HWC",
            Self::Chw => "CHW",
        }
    }

    pub fn batch_as_str(self) -> &'static str {
        match self {
            Self::Hwc => "NHWC",
            Self::Chw => "NCHW",
        }
    }
}

/// An encoded (still compressed) image sample.
///
/// `Buffer::clone()` retains the Arrow buffer allocation, so encoded sources
/// remain zero-copy at the persistence boundary.
#[derive(Clone, Debug)]
pub struct EncodedImageSample {
    pub image: Buffer,
    pub label: i64,
}

#[derive(Clone, Debug)]
pub struct DecodedSample {
    pub image: Tensor,
    pub label: i64,
}

#[derive(Clone, Debug)]
pub struct ImageBatch {
    pub images: Tensor,
    pub labels: Tensor,
}

#[derive(Clone, Debug)]
pub enum ImageSample {
    Encoded(EncodedImageSample),
    Decoded(DecodedSample),
}

impl ImageSample {
    pub fn into_decoded(self) -> RivetResult<DecodedSample> {
        match self {
            Self::Decoded(sample) => Ok(sample),
            Self::Encoded(_) => Err(invalid_argument("image must be decoded before batching")),
        }
    }
}
