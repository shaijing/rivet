use arrow_array::{Array, BinaryArray, StructArray};
use image::{DynamicImage, ImageReader};
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use std::io::Cursor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::open("/tmp/data.lance").await?;

    // Lance 11: 显式构造 projection
    let projection = ProjectionRequest::from_columns(["img"], dataset.schema());

    // 随机读取第 25000 行
    let batch = dataset.take(&[25_000], projection).await?;
    println!("batch schema = {:#?}", batch.schema());
    // img: struct<bytes: binary, path: string>
    let img_col = batch.column_by_name("img").ok_or("img column not found")?;
    println!("img datatype = {:?}", img_col.data_type());
    println!("img array = {:?}", img_col);
    let img_struct = img_col
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or("img is not StructArray")?;

    let bytes_col = img_struct
        .column_by_name("bytes")
        .ok_or("img.bytes not found")?;

    let bytes_array = bytes_col
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or("img.bytes is not BinaryArray")?;

    // zero-copy borrow：
    // 这里没有把 PNG bytes 复制到新的 Vec
    let encoded: &[u8] = bytes_array.value(0);

    println!("encoded len = {}", encoded.len());
    println!("signature = {:02X?}", &encoded[..8]);

    // &[u8] 直接送给 PNG decoder
    let image = ImageReader::new(Cursor::new(encoded))
        .with_guessed_format()?
        .decode()?;

    println!("decoded: {}x{}", image.width(), image.height());
    let pixels: Vec<u8> = match image {
        DynamicImage::ImageRgb8(img) => img.into_raw(),
        other => other.into_rgb8().into_raw(),
    };
    // decode 后的真实 RGB 像素
    // let rgb = image.to_rgb8();
    // let width = rgb.width();
    // let height = rgb.height();

    // let pixels: Vec<u8> = rgb.into_raw();

    // println!("shape = [{height}, {width}, 3]");
    println!("pixel bytes = {}", pixels.len());

    Ok(())
}
