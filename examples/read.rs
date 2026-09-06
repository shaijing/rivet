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
        "usage: cargo run --example read -- <arrow-file> or set RIVET_TEST_ARROW_FILE",
    )
    .into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(arrow_file_arg()?)?;

    let reader = StreamReader::try_new(file, None)?;

    for batch in reader {
        let batch = batch?;

        println!("rows: {}", batch.num_rows());
        println!("columns: {}", batch.num_columns());
        println!("schema: {:?}", batch.schema());
    }

    Ok(())
}
