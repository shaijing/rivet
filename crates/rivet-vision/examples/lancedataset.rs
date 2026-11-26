use arrow_array::{Array, BinaryArray, Int32Array, StringArray};
use image::ImageReader;
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use std::io::Cursor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or(
        "usage: cargo run -p rivet-vision --example lancedataset -- <physical-lance-path>",
    )?;
    let dataset = Dataset::open(&path).await?;

    // Rivet's converted cache requires image/label; custom conversions may
    // additionally preserve class names and source paths.
    let mut columns = vec!["image", "label"];
    for name in ["label_name", "path"] {
        if dataset.schema().field(name).is_some() {
            columns.push(name);
        }
    }
    let projection = ProjectionRequest::from_columns(columns, dataset.schema());

    let batch = dataset.take(&[1000], projection).await?;

    let image_col = batch
        .column_by_name("image")
        .ok_or("missing image column")?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or("image is not BinaryArray")?;

    let label_col = batch
        .column_by_name("label")
        .ok_or("missing label column")?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or("label is not Int32Array")?;

    // zero-copy borrow of encoded JPEG bytes
    let encoded: &[u8] = image_col.value(0);

    let label = label_col.value(0);
    println!("label: {label}");
    for name in ["label_name", "path"] {
        if let Some(column) = batch.column_by_name(name) {
            let column = column
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("metadata column is not StringArray")?;
            println!("{name}: {}", column.value(0));
        }
    }
    println!("encoded bytes: {}", encoded.len());
    println!("signature: {:02X?}", &encoded[..encoded.len().min(10)]);

    // JPEG decode
    let image = ImageReader::new(Cursor::new(encoded))
        .with_guessed_format()?
        .decode()?;

    let rgb = image.into_rgb8();

    let width = rgb.width();
    let height = rgb.height();

    // move out the underlying pixel Vec<u8>
    let pixels: Vec<u8> = rgb.into_raw();

    println!("decoded: {width}x{height}");
    println!("shape: [{height}, {width}, 3]");
    println!("pixel bytes: {}", pixels.len());

    Ok(())
}
