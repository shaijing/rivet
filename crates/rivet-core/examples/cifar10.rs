//! End-to-end CIFAR-10 example: mmap-backed Arrow dataset -> multi-worker
//! pipeline -> training-style batch loop over (x, y).
//!
//! ```text
//! cargo run -p rivet-core --example cifar10            # all batches
//! cargo run -p rivet-core --example cifar10 -- 10      # first 10 batches
//! ```
//!
//! The Arrow file comes from `RIVET_TEST_ARROW_FILE` or the local Hugging
//! Face CIFAR-10 cache (populated by `load_dataset("uoft-cs/cifar10")`);
//! the optional argument caps the number of batches.

use rivet_core::dataset::{ArrowImageDataset, Dataset};
use rivet_core::pipeline::ImagePipeline;
use rivet_core::runtime::ImageDataLoader;
use rivet_core::sample::image::{ImageBatch, ImageBuffer};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;

const IMAGE_COLUMN: &str = "img";
const LABEL_COLUMN: &str = "label";
const MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
const STD: [f32; 3] = [0.2470, 0.2435, 0.2616];

fn find_cache_file() -> Option<PathBuf> {
    let roots = [
        env::var("HF_DATASETS_CACHE").ok().map(PathBuf::from),
        env::var("HF_HOME").ok().map(|home| PathBuf::from(home).join("datasets")),
        env::var("HOME").ok().map(|home| {
            PathBuf::from(home).join(".cache/huggingface/datasets")
        }),
    ]
    .into_iter()
    .flatten();

    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let mut found = WalkDir::new(&root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name() == "cifar10-train.arrow")
            .map(|entry| entry.into_path())
            .collect::<Vec<_>>();
        if !found.is_empty() {
            found.sort();
            return found.pop();
        }
    }
    None
}

fn arrow_file_arg() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::var_os("RIVET_TEST_ARROW_FILE") {
        return Ok(PathBuf::from(path));
    }
    find_cache_file().ok_or_else(|| {
        std::io::Error::other(
            "no CIFAR-10 Arrow file found: set RIVET_TEST_ARROW_FILE or load \
             'uoft-cs/cifar10' with the datasets library first",
        )
        .into()
    })
}

/// A minimal "training step" over one batch, mirroring real use: `x` is the
/// normalized image tensor and `y` the labels.
fn train_step(step: usize, batch: ImageBatch) {
    let ImageBatch {
        images, labels, ..
    } = batch;
    let dtype = images.dtype().as_str();
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
}

fn batch_count(images: ImageBuffer) -> usize {
    match images {
        ImageBuffer::U8(values) => values.len() / (32 * 32 * 3),
        ImageBuffer::F32(values) => values.len() / (3 * 32 * 32),
    }
}

fn run(loader: &mut ImageDataLoader, max_batches: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut batches = 0usize;
    let mut images = 0usize;
    let mut first_label: Option<i64> = None;
    let mut last_label: Option<i64> = None;
    let start = Instant::now();

    while batches < max_batches {
        match loader.next_batch()? {
            Some(batch) => {
                batches += 1;
                images += batch.shape.0;
                last_label = batch.labels.last().copied();
                first_label.get_or_insert_with(|| batch.labels[0]);
                train_step(batches, batch);
            }
            None => break,
        }
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
    let plan = pipeline.clone().batch(1, false).compile()?.plan;
    println!("output state: {:?}", plan.output_state);

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
