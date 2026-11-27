use crate::errors::{DataResult, invalid_argument};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow_buffer::Buffer;
use arrow_ipc::convert::fb_to_schema;
use arrow_ipc::reader::{FileDecoder, StreamDecoder, read_footer_length};
use arrow_ipc::{Block, root_as_footer};
use bytes::Bytes;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Options for opening an Arrow IPC file. Extensible (alignment policy,
/// schema validation, mmap behavior, future IPC knobs) instead of a growing
/// list of bool parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArrowOpenOptions {
    /// `true`: fail on misaligned IPC buffers instead of silently copying
    /// them to an aligned heap buffer. Tests use this to prove the mmap
    /// path stays zero-copy.
    pub require_alignment: bool,
}

impl Default for ArrowOpenOptions {
    fn default() -> Self {
        Self {
            require_alignment: false,
        }
    }
}

/// One mmap-decoded IPC file or stream: the stream schema plus its record batches.
///
/// The schema is carried independently of the batches so that a valid
/// stream with a schema but zero record batches (an empty dataset) still
/// exposes its schema.
pub(crate) struct DecodedArrowFile {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// Decode an Arrow IPC file or stream into record batches whose array buffers
/// stay backed by a read-only mmap of the file.
///
/// Ownership chain: `Mmap` -> `Bytes::from_owner` -> `Buffer`. Every record
/// batch buffer sliced out by [`StreamDecoder`] shares that `Buffer`'s
/// allocation, so the batches keep the mmap alive via refcount without the
/// caller having to store the `Mmap` itself.
pub(crate) fn decode_mmap_arrow(
    path: &Path,
    options: &ArrowOpenOptions,
) -> DataResult<DecodedArrowFile> {
    let file = File::open(path)?;

    // SAFETY: the mapping is read-only. The mapped file must not be truncated
    // or written by others while the returned batches keep the mapping alive.
    let mmap = unsafe { Mmap::map(&file)? };

    let bytes = Bytes::from_owner(mmap);
    let mut buffer = Buffer::from(bytes);
    if buffer.starts_with(b"ARROW1") {
        return decode_file(&buffer, options);
    }
    let mut decoder = StreamDecoder::new().with_require_alignment(options.require_alignment);
    let mut batches = Vec::new();

    while !buffer.is_empty() {
        if let Some(batch) = decoder.decode(&mut buffer)? {
            batches.push(batch);
        }
    }
    decoder.finish()?;

    let schema = decoder
        .schema()
        .ok_or_else(|| invalid_argument(format!("arrow file {} has no schema", path.display())))?;

    Ok(DecodedArrowFile { schema, batches })
}

fn decode_file(buffer: &Buffer, options: &ArrowOpenOptions) -> DataResult<DecodedArrowFile> {
    let invalid_footer = || invalid_argument("invalid Arrow IPC file footer");
    let trailer = buffer.len().checked_sub(10).ok_or_else(invalid_footer)?;
    let footer_len =
        read_footer_length(buffer[trailer..].try_into().map_err(|_| invalid_footer())?)?;
    let start = trailer
        .checked_sub(footer_len)
        .filter(|&start| start >= 8)
        .ok_or_else(invalid_footer)?;
    let footer = root_as_footer(&buffer[start..trailer]).map_err(invalid_argument)?;
    let schema = std::sync::Arc::new(fb_to_schema(footer.schema().ok_or_else(invalid_footer)?));
    let mut decoder = FileDecoder::new(schema.clone(), footer.version())
        .with_require_alignment(options.require_alignment);
    let block_buffer = |block: &Block| {
        let invalid = || invalid_argument("Arrow IPC block extends outside the file body");
        let offset = usize::try_from(block.offset()).map_err(|_| invalid())?;
        let metadata = usize::try_from(block.metaDataLength()).map_err(|_| invalid())?;
        let body = usize::try_from(block.bodyLength()).map_err(|_| invalid())?;
        let len = metadata.checked_add(body).ok_or_else(invalid)?;
        if offset < 8 || offset.checked_add(len).is_none_or(|end| end > start) {
            return Err(invalid());
        }
        Ok(buffer.slice_with_length(offset, len))
    };
    for block in footer.dictionaries().iter().flatten() {
        decoder.read_dictionary(block, &block_buffer(block)?)?;
    }
    let mut batches = Vec::new();
    for block in footer.recordBatches().iter().flatten() {
        if let Some(batch) = decoder.read_record_batch(block, &block_buffer(block)?)? {
            batches.push(batch);
        }
    }
    Ok(DecodedArrowFile { schema, batches })
}
