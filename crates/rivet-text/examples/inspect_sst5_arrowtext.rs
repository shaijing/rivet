use arrow::array::{Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

const DEFAULT_DATA_DIR: &str = "/home/ling/.cache/huggingface/datasets/setfit___sst5";
const SPLITS: [&str; 3] = ["train", "validation", "test"];

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DATA_DIR));

    println!("ArrowText directory: {}", data_dir.display());
    for split in SPLITS {
        inspect_split(&data_dir, split)?;
    }
    Ok(())
}

fn inspect_split(data_dir: &Path, split: &str) -> Result<(), Box<dyn Error>> {
    let filename = format!("sst5-{split}.arrow");
    let path = find_file(data_dir, &filename)?
        .ok_or_else(|| format!("could not find {filename} under {}", data_dir.display()))?;
    let file = File::open(&path)?;
    let mut reader = StreamReader::try_new(file, None)?;
    let schema = reader.schema();
    let mut row_count = 0usize;
    let mut samples = Vec::new();
    let mut labels = BTreeMap::<i64, (String, usize)>::new();
    let mut token_lengths = Vec::new();

    for batch in reader.by_ref() {
        let batch = batch?;
        accumulate_batch(
            &batch,
            &mut row_count,
            &mut samples,
            &mut labels,
            &mut token_lengths,
        )?;
    }

    println!("\n[{split}] {}", path.display());
    println!("schema: {schema}");
    println!("rows: {row_count}");
    println!(
        "whitespace-token lengths: {}",
        length_summary(&mut token_lengths)
    );
    println!("label distribution:");
    for (label, (label_text, count)) in labels {
        println!("  {label}: {label_text:?} -> {count}");
    }
    println!("first {} rows:", samples.len());
    for (index, (text, label, label_text)) in samples.iter().enumerate() {
        println!("  {index}: label={label} ({label_text:?}), text={text:?}");
    }
    Ok(())
}

fn accumulate_batch(
    batch: &RecordBatch,
    row_count: &mut usize,
    samples: &mut Vec<(String, i64, String)>,
    labels: &mut BTreeMap<i64, (String, usize)>,
    token_lengths: &mut Vec<usize>,
) -> Result<(), Box<dyn Error>> {
    let text = batch
        .column_by_name("text")
        .ok_or("missing text column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("text column is not utf8")?;
    let label = batch
        .column_by_name("label")
        .ok_or("missing label column")?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or("label column is not int64")?;
    let label_text = batch
        .column_by_name("label_text")
        .ok_or("missing label_text column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("label_text column is not utf8")?;

    for row in 0..batch.num_rows() {
        if text.is_null(row) || label.is_null(row) || label_text.is_null(row) {
            return Err(format!("unexpected null at row {}", *row_count + row).into());
        }
        let text_value = text.value(row);
        let label_value = label.value(row);
        let label_text_value = label_text.value(row);
        let entry = labels
            .entry(label_value)
            .or_insert_with(|| (label_text_value.to_owned(), 0));
        entry.1 += 1;
        token_lengths.push(text_value.split_whitespace().count());
        if samples.len() < 3 {
            samples.push((
                text_value.to_owned(),
                label_value,
                label_text_value.to_owned(),
            ));
        }
    }
    *row_count += batch.num_rows();
    Ok(())
}

fn length_summary(lengths: &mut [usize]) -> String {
    if lengths.is_empty() {
        return "empty".to_owned();
    }
    lengths.sort_unstable();
    let total: usize = lengths.iter().sum();
    let percentile =
        |p: usize| lengths[((lengths.len() * p).div_ceil(100) - 1).min(lengths.len() - 1)];
    format!(
        "min={}, mean={:.1}, p50={}, p95={}, p99={}, max={}",
        lengths[0],
        total as f64 / lengths.len() as f64,
        percentile(50),
        percentile(95),
        percentile(99),
        lengths[lengths.len() - 1],
    )
}

fn find_file(root: &Path, filename: &str) -> io::Result<Option<PathBuf>> {
    let direct = root.join(filename);
    if direct.is_file() {
        return Ok(Some(direct));
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir()
            && let Some(found) = find_file(&path, filename)?
        {
            return Ok(Some(found));
        }
    }
    Ok(None)
}
