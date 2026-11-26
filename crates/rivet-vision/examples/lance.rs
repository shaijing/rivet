use futures::StreamExt;
use lance::Dataset;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: cargo run -p rivet-vision --example lance -- <physical-lance-path>")?;
    let dataset = Dataset::open(&path).await?;

    println!("schema:\n{:?}", dataset.schema());
    println!("rows: {}", dataset.count_rows(None).await?);

    let scanner = dataset.scan();
    let mut stream = scanner.try_into_stream().await?;

    while let Some(batch) = stream.next().await {
        let batch = batch?;

        println!(
            "batch rows={}, columns={}",
            batch.num_rows(),
            batch.num_columns()
        );

        println!("{batch:?}");

        // 这里只演示第一个 batch
        break;
    }

    Ok(())
}
