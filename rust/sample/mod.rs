use crate::errors::{RivetResult, invalid_argument};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageDType {
    U8,
    F32,
}

impl ImageDType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::F32 => "float32",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageLayout {
    Hwc,
    Chw,
}

impl ImageLayout {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hwc => "HWC",
            Self::Chw => "CHW",
        }
    }

    pub(crate) fn batch_as_str(self) -> &'static str {
        match self {
            Self::Hwc => "NHWC",
            Self::Chw => "NCHW",
        }
    }
}

pub(crate) enum ImageBuffer {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl ImageBuffer {
    pub(crate) fn dtype(&self) -> ImageDType {
        match self {
            Self::U8(_) => ImageDType::U8,
            Self::F32(_) => ImageDType::F32,
        }
    }
}

pub(crate) struct EncodedImageSample {
    pub(crate) image: Vec<u8>,
    pub(crate) label: i64,
}

pub(crate) struct DecodedSample {
    pub(crate) image: ImageBuffer,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u8,
    pub(crate) label: i64,
    pub(crate) layout: ImageLayout,
}

pub(crate) struct ImageBatch {
    pub(crate) images: ImageBuffer,
    pub(crate) labels: Vec<i64>,
    pub(crate) shape: (usize, usize, usize, usize),
    pub(crate) layout: ImageLayout,
}

pub(crate) enum ImageSample {
    Encoded(EncodedImageSample),
    Decoded(DecodedSample),
}

impl ImageSample {
    pub(crate) fn into_decoded(self) -> RivetResult<DecodedSample> {
        match self {
            Self::Decoded(sample) => Ok(sample),
            Self::Encoded(_) => Err(invalid_argument("image must be decoded before batching")),
        }
    }
}
