use arrow_ipc::reader::StreamReader;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(
        "/home/ling/.cache/huggingface/datasets/uoft-cs___cifar10/plain_text/0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2/cifar10-train.arrow",
    )?;

    let reader = StreamReader::try_new(file, None)?;

    for batch in reader {
        let batch = batch?;

        println!("rows: {}", batch.num_rows());
        println!("columns: {}", batch.num_columns());
        println!("schema: {:?}", batch.schema());
    }

    Ok(())
}
