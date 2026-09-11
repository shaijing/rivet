use crate::errors::{RivetResult, invalid_argument};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
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

/// One mmap-decoded IPC stream: the stream schema plus its record batches.
///
/// The schema is carried independently of the batches so that a valid
/// stream with a schema but zero record batches (an empty dataset) still
/// exposes its schema.
pub(crate) struct DecodedArrowFile {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// Decode an Arrow IPC stream file into record batches whose array buffers
/// stay backed by a read-only mmap of the file.
///
/// Ownership chain: `Mmap` -> `Bytes::from_owner` -> `Buffer`. Every record
/// batch buffer sliced out by [`StreamDecoder`] shares that `Buffer`'s
/// allocation, so the batches keep the mmap alive via refcount without the
/// caller having to store the `Mmap` itself.
pub(crate) fn decode_mmap_arrow(
    path: &Path,
    options: &ArrowOpenOptions,
) -> RivetResult<DecodedArrowFile> {
    let file = File::open(path)?;

    // SAFETY: the mapping is read-only. The mapped file must not be truncated
    // or written by others while the returned batches keep the mapping alive.
    let mmap = unsafe { Mmap::map(&file)? };

    let bytes = Bytes::from_owner(mmap);
    let mut buffer = Buffer::from(bytes);
    let mut decoder = StreamDecoder::new().with_require_alignment(options.require_alignment);
    let mut batches = Vec::new();

    while !buffer.is_empty() {
        if let Some(batch) = decoder.decode(&mut buffer)? {
            batches.push(batch);
        }
    }
    decoder.finish()?;

    let schema = decoder.schema().ok_or_else(|| {
        invalid_argument(format!("arrow file {} has no schema", path.display()))
    })?;

    Ok(DecodedArrowFile { schema, batches })
}
