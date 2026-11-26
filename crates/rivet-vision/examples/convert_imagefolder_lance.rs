//! Convert an ImageFolder tree into a Rivet Lance dataset.
//!
//! Expected input layout: `<input>/<split>/<class>/*.{jpg,jpeg,png,bmp,webp}`.
//! Usage:
//!   cargo run -j 12 -p rivet-vision --features lance \
//!     --example convert_imagefolder_lance -- <imagefolder-root> [output-root] [--overwrite]
//!
//! Example:
//!   cargo run -j 12 -p rivet-vision --features lance \
//!     --example convert_imagefolder_lance -- \
//!     ./nuimages_classification \
//!     ./nuimages_classification.lance

use rivet_vision::api::convert_imagefolder_dataset;
use std::env;
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage: convert_imagefolder_lance <imagefolder-root> [output-root] [--overwrite]"
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input_root = PathBuf::from(args.next().ok_or_else(|| usage().to_owned())?);
    let mut output_root = None;
    let mut overwrite = false;
    for arg in args {
        if arg == "--overwrite" {
            overwrite = true;
        } else if !arg.starts_with("--") && output_root.is_none() {
            output_root = Some(PathBuf::from(arg));
        } else {
            return Err(format!("unknown argument {arg}; {}", usage()).into());
        }
    }
    let output_root = output_root.unwrap_or_else(|| {
        let mut output = input_root.as_os_str().to_owned();
        output.push(".lance");
        PathBuf::from(output)
    });
    let report = convert_imagefolder_dataset(input_root, output_root, None, overwrite)?;
    for (split, rows) in report.rows_by_split {
        println!("{split}: {rows} images");
    }
    println!("wrote Rivet dataset: {}", report.output_root.display());
    Ok(())
}
