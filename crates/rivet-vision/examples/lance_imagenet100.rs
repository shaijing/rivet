//! Read the Rivet-native ImageNet-100 Lance dataset.
//!
//! Usage:
//!   cargo run -j 12 -p rivet-vision --features lance --example lance_imagenet100
//!   cargo run -j 12 -p rivet-vision --features lance --example lance_imagenet100 -- validation 2 32
//!
//! Arguments are: [split] [max_batches] [batch_size].
//! Defaults are: train 2 32.
//!
//! The dataset root can be overridden with RIVET_IMAGENET100_ROOT. By default
//! it reads ~/.cache/rivet/datasets/imagenet-100.

use rivet_data::dataset::DatasetLoadResult;
use rivet_vision::datasets::load_lance_image_dataset;
use rivet_vision::sample::image::ImageSample;
use rivet_vision::source::ImageSource;
use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

const DATASET_NAME: &str = "imagenet-100";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn dataset_root() -> PathBuf {
    if let Some(path) = env::var_os("RIVET_IMAGENET100_ROOT") {
        return PathBuf::from(path);
    }
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".cache/rivet/datasets").join(DATASET_NAME))
        .unwrap_or_else(|| PathBuf::from(".cache/rivet/datasets").join(DATASET_NAME))
}

fn parse_args() -> ExampleResult<(String, usize, usize)> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() > 3 {
        return Err(io::Error::other(
            "usage: lance_imagenet100 [split] [max_batches] [batch_size]",
        )
        .into());
    }

    let split = args.first().map(String::as_str).unwrap_or("train");
    if !matches!(split, "train" | "validation") {
        return Err(io::Error::other("split must be train or validation").into());
    }
    let max_batches = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("2")
        .parse::<usize>()
        .map_err(|_| io::Error::other("max_batches must be a positive integer"))?;
    let batch_size = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("32")
        .parse::<usize>()
        .map_err(|_| io::Error::other("batch_size must be a positive integer"))?;
    if max_batches == 0 || batch_size == 0 {
        return Err(io::Error::other("max_batches and batch_size must be positive").into());
    }

    Ok((split.to_owned(), max_batches, batch_size))
}

fn open_split(root: &std::path::Path, split: &str) -> ExampleResult<ImageSource> {
    let loaded = load_lance_image_dataset(root, "image", "label")?;
    match loaded {
        DatasetLoadResult::Single(_) => Err(io::Error::other(format!(
            "expected a dataset root containing splits at {}, but it resolved to one physical dataset",
            root.display()
        ))
        .into()),
        DatasetLoadResult::Bundle(bundle) => {
            let available = bundle.split_names().collect::<Vec<_>>().join(", ");
            let dataset = bundle.split(split).ok_or_else(|| {
                io::Error::other(format!(
                    "split '{split}' not found in {}; available splits: {available}",
                    root.display()
                ))
            })?;
            println!("available splits: {available}");
            Ok(dataset)
        }
    }
}

fn read_batches(dataset: &ImageSource, max_batches: usize, batch_size: usize) -> ExampleResult<()> {
    let row_count = dataset.len();
    let start = Instant::now();
    let mut rows_read = 0usize;

    for batch_index in 0..max_batches {
        let begin = batch_index.saturating_mul(batch_size);
        if begin >= row_count {
            break;
        }
        let end = (begin + batch_size).min(row_count);
        let indices: Vec<usize> = (begin..end).collect();
        let samples = dataset.get_many(&indices)?;
        if samples.len() != indices.len() {
            return Err(io::Error::other(format!(
                "Rivet returned {} samples for {} requested indices",
                samples.len(),
                indices.len()
            ))
            .into());
        }

        let first = match samples
            .first()
            .ok_or_else(|| io::Error::other("Rivet returned an empty batch"))?
        {
            ImageSample::Encoded(sample) => sample,
            ImageSample::Decoded(_) => {
                return Err(io::Error::other("expected an encoded Lance source").into());
            }
        };
        let image = image::load_from_memory(first.image.as_slice())?;
        rows_read += samples.len();
        println!(
            "batch {batch_index}: rows={} first_label={} image={}x{}",
            samples.len(),
            first.label,
            image.width(),
            image.height(),
        );
    }

    let elapsed = start.elapsed();
    println!(
        "read {rows_read} rows in {:.3}s ({:.1} rows/s)",
        elapsed.as_secs_f64(),
        rows_read as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
    );
    Ok(())
}

fn main() -> ExampleResult<()> {
    let (split, max_batches, batch_size) = parse_args()?;
    let root = dataset_root();
    println!("opening ImageNet-100 split={split} at {}", root.display());
    let dataset = open_split(&root, &split)?;
    println!(
        "rows: {}, batch_size: {}, max_batches: {}",
        dataset.len(),
        batch_size,
        max_batches
    );
    read_batches(&dataset, max_batches, batch_size)
}
