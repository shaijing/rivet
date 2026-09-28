//! Convert a local Hugging Face Arrow cache to Rivet-native Lance datasets.
//!
//! Usage:
//!   cargo run -j 12 -p rivet-vision --features lance \
//!     --example convert_hf_lance -- <hf-arrow-root> [output-root]
//!
//! Optional flags:
//!   --image-column <name>  (default: img)
//!   --label-column <name>  (default: label)
//!   --keep-path
//!   --overwrite
//!   --max-bytes-per-file <bytes>  (default: 2147483648)
//!   --max-rows-per-file <rows>  (default: 1048576)

use rivet_vision::datasets::lance::{
    HuggingFaceLanceOptions, convert_huggingface_dataset, discover_huggingface_splits,
};
use std::env;
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage: convert_hf_lance <hf-arrow-root-or-file> [output-root] [options]"
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().ok_or_else(|| usage().to_owned())?);
    let mut output = None;
    let mut options = HuggingFaceLanceOptions::default();

    while let Some(option) = args.next() {
        match option.as_str() {
            value if !value.starts_with("--") && output.is_none() => {
                output = Some(PathBuf::from(value));
            }
            "--image-column" => {
                options.image_column = args.next().ok_or("--image-column requires a value")?;
            }
            "--label-column" => {
                options.label_column = args.next().ok_or("--label-column requires a value")?;
            }
            "--keep-path" => options.keep_path = true,
            "--overwrite" => options.overwrite = true,
            "--max-bytes-per-file" => {
                let value = args.next().ok_or("--max-bytes-per-file requires a value")?;
                options.max_bytes_per_file = value.parse().map_err(|error| {
                    format!("invalid --max-bytes-per-file value '{value}': {error}")
                })?;
            }
            "--max-rows-per-file" => {
                let value = args.next().ok_or("--max-rows-per-file requires a value")?;
                options.max_rows_per_file = value.parse().map_err(|error| {
                    format!("invalid --max-rows-per-file value '{value}': {error}")
                })?;
            }
            _ => return Err(format!("unknown option {option}; {}", usage()).into()),
        }
    }

    let output = output.unwrap_or_else(|| default_output_root(&input));
    let splits = discover_huggingface_splits(&input)?;
    let report = convert_huggingface_dataset(splits, output, &options)?;
    for (split, rows) in report.rows_by_split {
        println!("{split}: {rows} rows");
    }
    println!("wrote Rivet dataset: {}", report.output_root.display());
    Ok(())
}

fn default_output_root(input: &std::path::Path) -> PathBuf {
    let raw_name = input
        .file_stem()
        .or_else(|| input.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("dataset");
    let dataset_name = raw_name
        .rsplit_once("___")
        .map(|(_, name)| name)
        .unwrap_or(raw_name);
    let base = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".cache/rivet/datasets"))
        .unwrap_or_else(|| PathBuf::from(".cache/rivet/datasets"));
    base.join(dataset_name)
}
