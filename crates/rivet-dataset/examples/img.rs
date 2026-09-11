use arrow::array::{Array, BinaryArray, Int64Array, StructArray};
use arrow_ipc::reader::StreamReader;
use std::env;
use std::fs::File;
use std::io;
use std::path::PathBuf;

fn arrow_file_arg() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::args_os().nth(1) {
        return Ok(path.into());
    }
    if let Some(path) = env::var_os("RIVET_TEST_ARROW_FILE") {
        return Ok(path.into());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: cargo run --example img -- <arrow-file> or set RIVET_TEST_ARROW_FILE",
    )
    .into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(arrow_file_arg()?)?;

    let mut reader = StreamReader::try_new(file, None)?;

    let batch = reader.next().unwrap()?;

    // img
    let img_index = batch.schema().index_of("img")?;
    let img = batch
        .column(img_index)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("img must be StructArray");

    // img.bytes
    let bytes = img
        .column_by_name("bytes")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("img.bytes must be BinaryArray");

    // label
    let label_index = batch.schema().index_of("label")?;
    let labels = batch
        .column(label_index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("label must be Int64Array");

    for row in 0..batch.num_rows().min(10) {
        if !bytes.is_null(row) {
            let encoded = bytes.value(row);
            let label = labels.value(row);

            println!(
                "row={row}, label={label}, encoded_size={} bytes",
                encoded.len()
            );
            let encoded = bytes.value(row);
            let img = image::load_from_memory(encoded)?;

            println!(
                "row={row}, label={label}, size={}x{}, color={:?}",
                img.width(),
                img.height(),
                img.color()
            );
        }
    }

    Ok(())
}
