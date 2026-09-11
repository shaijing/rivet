use arrow::array::{Array, BinaryArray, StructArray};
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamDecoder;
use bytes::Bytes;
use memmap2::Mmap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

fn mmap_arrow_file(path: impl AsRef<Path>) -> Result<Buffer, Box<dyn Error>> {
    let file = File::open(path)?;

    // SAFETY:
    // 只读 mmap，且假设文件在映射生命周期内不会被外部修改。
    let mmap = unsafe { Mmap::map(&file)? };

    println!("mmap len: {} bytes", mmap.len());
    println!(
        "mmap range: {:p} .. {:p}",
        mmap.as_ptr(),
        unsafe { mmap.as_ptr().add(mmap.len()) }
    );

    // zero-copy:
    // Bytes 持有 Mmap ownership
    let bytes = Bytes::from_owner(mmap);

    // zero-copy:
    // Arrow Buffer 复用 Bytes backing storage
    let buffer = Buffer::from(bytes);

    println!("buffer len: {} bytes", buffer.len());
    println!("buffer address: {:p}", buffer.as_ptr());

    Ok(buffer)
}

fn inspect_first_image_batch(
    batch: &arrow::record_batch::RecordBatch,
    mmap_start: usize,
    mmap_end: usize,
) -> Result<(), Box<dyn Error>> {
    println!();
    println!("=== inspect first batch ===");
    println!("rows: {}", batch.num_rows());
    println!("columns: {}", batch.num_columns());
    println!("schema: {}", batch.schema());

    let img_column = batch
        .column_by_name("img")
        .ok_or("missing img column")?;

    let img = img_column
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or("img column is not StructArray")?;

    let bytes_column = img
        .column_by_name("bytes")
        .ok_or("missing img.bytes field")?;

    let bytes = bytes_column
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or("img.bytes is not BinaryArray")?;

    println!("image count in batch: {}", bytes.len());

    let data = bytes.to_data();

    println!();
    println!("BinaryArray buffers:");

    for (i, buf) in data.buffers().iter().enumerate() {
        let start = buf.as_ptr() as usize;
        let end = start + buf.len();

        let fully_inside_mmap =
            start >= mmap_start && end <= mmap_end;

        println!(
            "buffer[{i}]: addr={:#x}, len={}, end={:#x}, in_mmap={}",
            start,
            buf.len(),
            end,
            fully_inside_mmap,
        );
    }

    if !bytes.is_empty() {
        let first = bytes.value(0);

        println!();
        println!("first image encoded size: {} bytes", first.len());

        let preview_len = first.len().min(16);

        println!(
            "first image first {} bytes: {:02x?}",
            preview_len,
            &first[..preview_len]
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = "/home/ling/.cache/huggingface/datasets/uoft-cs___cifar10/plain_text/0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2/cifar10-train.arrow";

    let mut buffer = mmap_arrow_file(path)?;

    // 保存完整 mmap 的原始地址范围。
    //
    // 注意 StreamDecoder::decode(&mut buffer) 会消费 buffer，
    // 使 buffer 本身不断 slice 向后移动。
    let mmap_start = buffer.as_ptr() as usize;
    let mmap_end = mmap_start + buffer.len();

    println!();
    println!(
        "saved mmap range: {:#x} .. {:#x}",
        mmap_start, mmap_end
    );

    let mut decoder = StreamDecoder::new();

    let mut batch_count = 0usize;
    let mut row_count = 0usize;
    let mut inspected = false;

    while !buffer.is_empty() {
        match decoder.decode(&mut buffer)? {
            Some(batch) => {
                batch_count += 1;
                row_count += batch.num_rows();

                if !inspected {
                    inspect_first_image_batch(
                        &batch,
                        mmap_start,
                        mmap_end,
                    )?;
                    inspected = true;
                }

                if batch_count <= 5 || batch_count % 100 == 0 {
                    println!(
                        "batch {:>3}: rows={}, columns={}",
                        batch_count,
                        batch.num_rows(),
                        batch.num_columns()
                    );
                }
            }

            None => {
                // Decoder 可能只消费了 schema / dictionary / metadata，
                // 此时没有产生 RecordBatch，继续即可。
            }
        }
    }

    decoder.finish()?;

    println!();
    println!("=== summary ===");
    println!("batches: {}", batch_count);
    println!("rows: {}", row_count);

    Ok(())
}