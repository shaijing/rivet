//! End-to-end ImageFolder example over the nuImages classification dataset.
//!
//! ```text
//! cargo run -p rivet-vision --example imagefolder -- ./nuimages_classification
//! cargo run -p rivet-vision --example imagefolder -- ./nuimages_classification val 1
//! ```
//!
//! The dataset has the usual split/class directory layout:
//!
//! ```text
//! ./nuimages_classification/
//! ├── train/<class>/*.jpg
//! └── val/<class>/*.jpg
//! ```

use rivet_data::dataset::Dataset;
use rivet_vision::datasets::ImageFolderDatasetCore;
use rivet_vision::pipeline::ImagePipeline;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

const IMAGE_SIZE: u32 = 224;
const BATCH_SIZE: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .ok_or("usage: imagefolder <root> [split] [max_batches]")?,
    );
    let split = args.next().unwrap_or_else(|| "train".to_string());
    let max_batches = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2);

    let root = root.join(&split);
    let dataset = Arc::new(ImageFolderDatasetCore::new(root.clone())?);

    println!("root: {}", root.display());
    println!("dataset: {} images", dataset.len());
    println!(
        "classes ({}): {:?}",
        dataset.classes().len(),
        dataset.classes()
    );
    println!("class_to_idx: {:?}", dataset.class_to_idx());

    let first = dataset
        .samples()
        .first()
        .ok_or("ImageFolder dataset is empty")?;
    let encoded = dataset.get(0)?;
    let decoded = image::load_from_memory(encoded.image.as_slice())?;
    println!(
        "first sample: label={} class={} path={} encoded={} bytes size={}x{}",
        encoded.label,
        dataset.classes()[encoded.label as usize],
        first.path.display(),
        encoded.image.len(),
        decoded.width(),
        decoded.height(),
    );

    let mut loader = ImagePipeline::new(Arc::clone(&dataset))
        .decode_image()
        .resize(IMAGE_SIZE, IMAGE_SIZE)
        .normalize(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
        .hwc_to_chw()
        .workers(4)
        .prefetch_batches(2)
        .batch(BATCH_SIZE, false)
        .compile()?;

    println!(
        "pipeline: decode -> resize({IMAGE_SIZE}x{IMAGE_SIZE}) -> normalize -> CHW; \
         workers=4 prefetch=2 batch={BATCH_SIZE}"
    );

    let mut batches = 0usize;
    let mut images = 0usize;
    for batch in (&mut loader).into_iter().take(max_batches) {
        let batch = batch?;
        assert_eq!(batch.images.dims()[1], 3);
        assert_eq!(batch.images.dims()[2], IMAGE_SIZE as usize);
        assert_eq!(batch.images.dims()[3], IMAGE_SIZE as usize);
        assert_eq!(format!("{:?}", batch.images.dtype()), "F32");

        batches += 1;
        images += batch.images.dims()[0];
        println!(
            "batch {batches}: shape={:?} dtype={} labels={:?}",
            batch.images.dims(),
            format!("{:?}", batch.images.dtype()),
            batch.labels.to_vec::<i64>()?,
        );
    }

    println!("done: {batches} batches, {images} images");
    Ok(())
}
