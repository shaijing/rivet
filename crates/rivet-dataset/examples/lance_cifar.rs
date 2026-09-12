//! Read the Rivet-native CIFAR Lance datasets through Rivet's API.
//!
//! Usage:
//!   cargo run -j 16 -p rivet-dataset --example lance_cifar
//!   cargo run -j 16 -p rivet-dataset --example lance_cifar -- cifar100 test 4 128
//!
//! Arguments are: <dataset> <split> <max_batches> <batch_size>.
//! Defaults are: cifar10 train 2 64.
//!
//! The dataset root can be overridden with RIVET_LANCE_ROOT:
//!   RIVET_LANCE_ROOT=/data/datasets/rivet cargo run -j 16 \
//!     -p rivet-dataset --example lance_cifar

use rivet_dataset::dataset::{DatasetLoadResult, ImageSource, load_lance_image_dataset};
use rivet_dataset::sample::image::ImageSample;
use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_DATASET_ROOT: &str = "/data/datasets/rivet";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct CifarSpec {
    name: &'static str,
    image_column: &'static str,
    label_column: &'static str,
}

impl CifarSpec {
    fn from_name(name: &str) -> ExampleResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "cifar10" => Ok(Self {
                name: "cifar10",
                image_column: "image",
                label_column: "label",
            }),
            "cifar100" => Ok(Self {
                name: "cifar100",
                image_column: "image",
                label_column: "label",
            }),
            _ => Err(io::Error::other("dataset must be cifar10 or cifar100").into()),
        }
    }
}

fn parse_args() -> ExampleResult<(CifarSpec, String, usize, usize)> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() > 4 {
        return Err(io::Error::other(
            "usage: lance_cifar [dataset] [split] [max_batches] [batch_size]",
        )
        .into());
    }

    let dataset = args.first().map(String::as_str).unwrap_or("cifar10");
    let split = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("train")
        .to_owned();
    let max_batches = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("2")
        .parse::<usize>()
        .map_err(|_| io::Error::other("max_batches must be a positive integer"))?;
    let batch_size = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("64")
        .parse::<usize>()
        .map_err(|_| io::Error::other("batch_size must be a positive integer"))?;

    if !matches!(split.as_str(), "train" | "test") {
        return Err(io::Error::other("split must be train or test").into());
    }
    if max_batches == 0 || batch_size == 0 {
        return Err(io::Error::other("max_batches and batch_size must be positive").into());
    }

    Ok((
        CifarSpec::from_name(dataset)?,
        split,
        max_batches,
        batch_size,
    ))
}

fn dataset_root() -> PathBuf {
    env::var_os("RIVET_LANCE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DATASET_ROOT))
}

fn open_split(root: &Path, spec: &CifarSpec, split: &str) -> ExampleResult<ImageSource> {
    let path = root.join(spec.name);
    let loaded = load_lance_image_dataset(&path, spec.image_column, spec.label_column)?;

    match loaded {
        DatasetLoadResult::Single(_) => Err(io::Error::other(format!(
            "expected a split root at {}, but it resolved to one physical dataset",
            path.display()
        ))
        .into()),
        DatasetLoadResult::Bundle(bundle) => {
            let available = bundle.split_names().collect::<Vec<_>>().join(", ");
            let dataset = bundle.split(split).ok_or_else(|| {
                io::Error::other(format!(
                    "split '{split}' not found in {}; available splits: {available}",
                    path.display()
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
    let (spec, split, max_batches, batch_size) = parse_args()?;
    let root = dataset_root();
    let dataset_root = root.join(spec.name);

    println!(
        "opening {} split={} at {}",
        spec.name,
        split,
        dataset_root.display()
    );
    let dataset = open_split(&root, &spec, &split)?;
    println!(
        "columns: image={}, label={}",
        spec.image_column, spec.label_column
    );
    println!(
        "rows: {}, batch_size: {}, max_batches: {}",
        dataset.len(),
        batch_size,
        max_batches
    );

    read_batches(&dataset, max_batches, batch_size)
}
