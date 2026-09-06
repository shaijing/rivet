use crate::errors::value_err;
use pyo3::prelude::*;

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

    pub(crate) fn bytes_per_value(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F32 => 4,
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

pub(crate) struct EncodedImageSample {
    pub(crate) image: Vec<u8>,
    pub(crate) label: i64,
}

pub(crate) struct DecodedSample {
    pub(crate) image: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u8,
    pub(crate) label: i64,
    pub(crate) dtype: ImageDType,
    pub(crate) layout: ImageLayout,
}

pub(crate) struct ImageBatch {
    pub(crate) images: Vec<u8>,
    pub(crate) labels: Vec<i64>,
    pub(crate) shape: (usize, usize, usize, usize),
    pub(crate) dtype: ImageDType,
    pub(crate) layout: ImageLayout,
}

pub(crate) enum ImageSample {
    Encoded(EncodedImageSample),
    Decoded(DecodedSample),
}

impl ImageSample {
    pub(crate) fn into_decoded(self) -> PyResult<DecodedSample> {
        match self {
            Self::Decoded(sample) => Ok(sample),
            Self::Encoded(_) => Err(value_err("image must be decoded before batching")),
        }
    }
}
