//! End-to-end CIFAR-10 example: mmap-backed Arrow dataset -> multi-worker
//! pipeline -> training-style batch loop over (x, y).
//!
//! ```text
//! RIVET_TEST_ARROW_FILE=./cifar10-train.arrow cargo run -p rivet-vision --example cifar10 -- 10
//! ```
//!
//! Set RIVET_TEST_ARROW_FILE to an Arrow IPC stream. The optional argument
//! limits the number of batches. For cached Lance data, use lance_cifar.

use rivet_data::dataset::Dataset;
use rivet_vision::datasets::ArrowImageDataset;
use rivet_vision::pipeline::ImagePipeline;
use rivet_vision::runtime::ImageDataLoader;
use rivet_vision::sample::image::ImageBatch;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const IMAGE_COLUMN: &str = "img";
const LABEL_COLUMN: &str = "label";
const MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
const STD: [f32; 3] = [0.2470, 0.2435, 0.2616];

fn arrow_file_arg() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::var_os("RIVET_TEST_ARROW_FILE") {
        return Ok(PathBuf::from(path));
    }
    Err("set RIVET_TEST_ARROW_FILE to a CIFAR-10 Arrow IPC stream".into())
}

/// A minimal "training step" over one batch, mirroring real use: `x` is the
/// normalized image tensor and `y` the labels.
fn train_step(step: usize, batch: ImageBatch) -> Result<(), Box<dyn std::error::Error>> {
    let ImageBatch { images, labels, .. } = batch;
    let dtype = format!("{:?}", images.dtype());
    let labels = labels.to_vec::<i64>()?;
    let (first, last) = match labels.as_slice() {
        [] => (None, None),
        [head, .., tail] => (Some(*head), Some(*tail)),
        [only] => (Some(*only), Some(*only)),
    };
    println!(
        "step {step}: x dtype={dtype} samples={} | y {} labels first={} last={}",
        batch_count(images),
        labels.len(),
        first.unwrap_or(-1),
        last.unwrap_or(-1),
    );
    Ok(())
}

fn batch_count(images: rivet_core::Tensor) -> usize {
    images.dims()[0]
}

fn run(loader: &mut ImageDataLoader, max_batches: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut batches = 0usize;
    let mut images = 0usize;
    let mut first_label: Option<i64> = None;
    let mut last_label: Option<i64> = None;
    let start = Instant::now();

    for batch in loader.into_iter().take(max_batches) {
        let batch = batch?;
        batches += 1;
        images += batch.images.dims()[0];
        let labels = batch.labels.to_vec::<i64>()?;
        last_label = labels.last().copied();
        first_label.get_or_insert_with(|| labels[0]);
        train_step(batches, batch)?;
    }

    let elapsed = start.elapsed();
    println!(
        "done: {batches} batches, {images} images, y first={} last={} in {:.2}s ({:.0} img/s)",
        first_label.unwrap_or(-1),
        last_label.unwrap_or(-1),
        elapsed.as_secs_f64(),
        images as f64 / elapsed.as_secs_f64(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let max_batches: Option<usize> = env::args_os()
        .nth(1)
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse().ok());
    let path = arrow_file_arg()?;
    println!("arrow file: {}", path.display());

    let dataset = Arc::new(ArrowImageDataset::new(
        vec![path],
        IMAGE_COLUMN.to_string(),
        LABEL_COLUMN.to_string(),
    )?);
    let sample = dataset.get(0)?;
    println!(
        "dataset: {} rows | first sample label={} encoded={} bytes",
        dataset.len(),
        sample.label,
        sample.image.len(),
    );

    let pipeline = ImagePipeline::new(Arc::clone(&dataset))
        .decode_image()
        .normalize(MEAN.to_vec(), STD.to_vec())
        .hwc_to_chw();

    // Compiled output state is known before any sample is processed.
    let info = pipeline.clone().batch(1, false).compile()?.info;
    println!("output state: {:?}", info.output_state);

    // Training-style loop with a persistent 4-worker pool and 2 batches of
    // prefetch.
    let mut loader = pipeline
        .workers(4)
        .prefetch_batches(2)
        .batch(256, false)
        .compile()?;
    println!("batch loop: workers=4 prefetch=2 batch=256");
    run(&mut loader, max_batches.unwrap_or(usize::MAX))
}
