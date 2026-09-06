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
}

pub(crate) struct ImageBatch {
    pub(crate) images: Vec<u8>,
    pub(crate) labels: Vec<i64>,
    pub(crate) shape: (usize, usize, usize, usize),
}
