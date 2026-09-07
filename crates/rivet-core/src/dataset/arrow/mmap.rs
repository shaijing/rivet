use arrow::record_batch::RecordBatch;
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
use bytes::Bytes;
use crate::errors::RivetResult;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Decode an Arrow IPC stream file into record batches whose array buffers
/// stay backed by a read-only mmap of the file.
///
/// Ownership chain: `Mmap` -> `Bytes::from_owner` -> `Buffer`. Every record
/// batch buffer sliced out by [`StreamDecoder`] shares that `Buffer`'s
/// allocation, so the batches keep the mmap alive via refcount without the
/// caller having to store the `Mmap` itself.
///
/// `require_alignment: true` makes the decoder fail on misaligned IPC
/// buffers instead of silently copying them to an aligned heap buffer;
/// tests use it to prove the mmap path stays zero-copy.
pub(crate) fn decode_mmap_arrow(
    path: &Path,
    require_alignment: bool,
) -> RivetResult<Vec<RecordBatch>> {
    let file = File::open(path)?;

    // SAFETY: the mapping is read-only. The mapped file must not be truncated
    // or written by others while the returned batches keep the mapping alive.
    let mmap = unsafe { Mmap::map(&file)? };

    let bytes = Bytes::from_owner(mmap);
    let mut buffer = Buffer::from(bytes);
    let mut decoder = StreamDecoder::new().with_require_alignment(require_alignment);
    let mut batches = Vec::new();

    while !buffer.is_empty() {
        if let Some(batch) = decoder.decode(&mut buffer)? {
            batches.push(batch);
        }
    }
    decoder.finish()?;

    Ok(batches)
}
