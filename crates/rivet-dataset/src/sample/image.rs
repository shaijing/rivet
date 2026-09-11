use arrow_buffer::Buffer;

use crate::errors::{RivetResult, invalid_argument};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageDType {
    U8,
    F32,
}

impl ImageDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::F32 => "float32",
        }
    }
}

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

pub enum ImageBuffer {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl ImageBuffer {
    pub fn dtype(&self) -> ImageDType {
        match self {
            Self::U8(_) => ImageDType::U8,
            Self::F32(_) => ImageDType::F32,
        }
    }
}

/// An encoded (still compressed) image sample.
///
/// `image` is an immutable binary payload that sources may back with an mmap,
/// a `Vec<u8>`, network `Bytes`, or any other shared allocation. The pipeline
/// only borrows it as `&[u8]` for decoding; no encoded-byte copy happens
/// inside the source boundary, and `Buffer::clone()` is a cheap shared
/// refcount bump.
pub struct EncodedImageSample {
    pub image: Buffer,
    pub label: i64,
}

pub struct DecodedSample {
    pub image: ImageBuffer,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub label: i64,
    pub layout: ImageLayout,
}

pub struct ImageBatch {
    pub images: ImageBuffer,
    pub labels: Vec<i64>,
    pub shape: (usize, usize, usize, usize),
    pub layout: ImageLayout,
}

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
