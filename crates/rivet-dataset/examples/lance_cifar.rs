//! Read the CIFAR Lance datasets stored under /data/datasets/rivet.
//!
//! Usage:
//!   cargo run -j 16 -p rivet-dataset --example lance_cifar
//!   cargo run -j 16 -p rivet-dataset --example lance_cifar -- cifar100 test 4 128
//!
//! Arguments are: <dataset> <split> <max_batches> <batch_size>.
//! Defaults are: cifar10 train 2 64.

use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array, RecordBatch, StructArray};
use lance::dataset::ProjectionRequest;
use lance::Dataset;
use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DATASET_ROOT: &str = "/data/datasets/rivet";

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct CifarSpec {
    name: &'static str,
    image_column: &'static str,
    label_column: &'static str,
    coarse_label_column: Option<&'static str>,
}

impl CifarSpec {
    fn from_name(name: &str) -> ExampleResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "cifar10" => Ok(Self {
                name: "cifar10",
                image_column: "img",
                label_column: "label",
                coarse_label_column: None,
            }),
            "cifar100" => Ok(Self {
                name: "cifar100",
                image_column: "img",
                label_column: "fine_label",
                coarse_label_column: Some("coarse_label"),
            }),
            _ => Err(io::Error::other("dataset must be cifar10 or cifar100").into()),
        }
    }
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> ExampleResult<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .ok_or_else(|| io::Error::other(format!("column {name} is missing")).into())
}

fn image_bytes<'a>(batch: &'a RecordBatch, name: &str, row: usize) -> ExampleResult<&'a [u8]> {
    let image = column(batch, name)?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| io::Error::other(format!("{name} is not a struct column")))?;
    let bytes = image
        .column_by_name("bytes")
        .ok_or_else(|| io::Error::other("img.bytes is missing"))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| io::Error::other("img.bytes is not a binary column"))?;

    if bytes.is_null(row) {
        return Err(io::Error::other(format!("{name}.bytes is null at row {row}")).into());
    }
    Ok(bytes.value(row))
}

fn labels<'a>(batch: &'a RecordBatch, name: &str) -> ExampleResult<&'a Int64Array> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| io::Error::other(format!("{name} is not an int64 column")).into())
}

fn parse_args() -> ExampleResult<(CifarSpec, String, usize, usize)> {
    let args: Vec<String> = env::args().skip(1).collect();
    let dataset = args.first().map(String::as_str).unwrap_or("cifar10");
    let split = args.get(1).map(String::as_str).unwrap_or("train").to_owned();
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

    Ok((CifarSpec::from_name(dataset)?, split, max_batches, batch_size))
}

fn dataset_path(spec: &CifarSpec, split: &str) -> PathBuf {
    Path::new(DATASET_ROOT)
        .join(spec.name)
        .join(format!("{split}.lance"))
}

async fn read_batches(
    dataset: &Dataset,
    spec: &CifarSpec,
    row_count: usize,
    max_batches: usize,
    batch_size: usize,
) -> ExampleResult<()> {
    let projection_columns = std::iter::once(spec.image_column)
        .chain(std::iter::once(spec.label_column))
        .chain(spec.coarse_label_column);
    let projection = ProjectionRequest::from_columns(projection_columns, dataset.schema());
    let start = Instant::now();
    let mut rows_read = 0usize;

    for batch_index in 0..max_batches {
        let begin = batch_index * batch_size;
        if begin >= row_count {
            break;
        }
        let end = (begin + batch_size).min(row_count);
        let indices: Vec<u64> = (begin..end).map(|index| index as u64).collect();
        let batch = dataset.take(&indices, projection.clone()).await?;

        if batch.num_rows() != indices.len() {
            return Err(io::Error::other(format!(
                "Lance returned {} rows for {} requested indices",
                batch.num_rows(),
                indices.len()
            ))
            .into());
        }

        let label_array = labels(&batch, spec.label_column)?;
        let first_image = image_bytes(&batch, spec.image_column, 0)?;
        let image = image::load_from_memory(first_image)?;
        rows_read += batch.num_rows();

        let coarse = spec
            .coarse_label_column
            .map(|name| labels(&batch, name).map(|array| array.value(0)))
            .transpose()?;

        println!(
            "batch {batch_index}: rows={} first_label={}{} image={}x{}",
            batch.num_rows(),
            label_array.value(0),
            coarse
                .map(|label| format!(" first_coarse_label={label}"))
                .unwrap_or_default(),
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

#[tokio::main]
async fn main() -> ExampleResult<()> {
    let (spec, split, max_batches, batch_size) = parse_args()?;
    let path = dataset_path(&spec, &split);

    println!("opening {} at {}", spec.name, path.display());
    let dataset = Dataset::open(path.to_str().ok_or_else(|| {
        io::Error::other(format!("dataset path is not valid UTF-8: {}", path.display()))
    })?)
    .await?;
    let row_count = dataset.count_rows(None).await?;
    println!("schema: {}", dataset.schema());
    println!("rows: {row_count}, batch_size: {batch_size}, max_batches: {max_batches}");

    read_batches(&dataset, &spec, row_count, max_batches, batch_size).await
}
