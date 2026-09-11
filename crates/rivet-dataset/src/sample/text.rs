use arrow_buffer::Buffer;

/// A raw, still un-tokenized UTF-8 text sample.
///
/// `text` holds UTF-8 bytes in an immutable [`Buffer`] so Arrow mmap
/// sources can hand over zero-copy slices of the column values buffer;
/// decode with `std::str::from_utf8(sample.text.as_slice())`.
pub struct RawTextSample {
    pub text: Buffer,
    pub label: Option<i64>,
}
