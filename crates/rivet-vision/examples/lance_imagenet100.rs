//! Read the Rivet-native ImageNet-100 Lance dataset.
//!
//! Usage:
//! cargo run -j 12 -p rivet-vision --example lance_imagenet100 -- ./imagenet-100 validation 2 32
//!
//! The source is a local Lance bundle or a remote source containing one.
//! With RIVET_IMAGENET100_ROOT set, omit the source argument.
//! Optional split, batch count, and batch size default to train, 2, and 32.

use rivet_data::dataset::DatasetLoadResult;
use rivet_vision::api::{LoadDatasetOptions, load_dataset};
use rivet_vision::sample::image::ImageSample;
use rivet_vision::source::ImageSource;
use std::env;
use std::error::Error;
use std::io;
use std::time::Instant;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn parse_args(args: Vec<String>) -> ExampleResult<(String, usize, usize)> {
    if args.len() > 3 {
        return Err(io::Error::other(
            "usage: lance_imagenet100 <source> [split] [max_batches] [batch_size]",
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
    let mut args = env::args().skip(1);
    let source = env::var("RIVET_IMAGENET100_ROOT")
        .ok()
        .or_else(|| args.next())
        .ok_or("usage: lance_imagenet100 <source> [split] [max_batches] [batch_size]")?;
    let (split, max_batches, batch_size) = parse_args(args.collect())?;
    println!("opening ImageNet-100 split={split} from {source}");
    let options = LoadDatasetOptions {
        split: Some(split),
        ..LoadDatasetOptions::default()
    };
    let DatasetLoadResult::Single(dataset) = load_dataset(source.as_str(), &options)? else {
        unreachable!("a specific split was requested");
    };
    println!(
        "rows: {}, batch_size: {}, max_batches: {}",
        dataset.len(),
        batch_size,
        max_batches
    );
    read_batches(&dataset, max_batches, batch_size)
}
