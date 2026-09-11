use arrow_array::{Array, BinaryArray, Int32Array, StringArray};
use image::ImageReader;
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use std::io::Cursor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::open("/data/datasets/custom/nuimages/train.lance").await?;

    let projection =
        ProjectionRequest::from_columns(["image", "label", "label_name", "path"], dataset.schema());

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

    let label_name_col = batch
        .column_by_name("label_name")
        .ok_or("missing label_name column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("label_name is not StringArray")?;

    let path_col = batch
        .column_by_name("path")
        .ok_or("missing path column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("path is not StringArray")?;

    // zero-copy borrow of encoded JPEG bytes
    let encoded: &[u8] = image_col.value(0);

    let label = label_col.value(0);
    let label_name = label_name_col.value(0);
    let path = path_col.value(0);

    println!("label: {label}");
    println!("label_name: {label_name}");
    println!("path: {path}");
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
