//! Write staged torchvision pixel arrays as losslessly encoded Lance images.

use super::convert::{native_schema, validate_split_name, write_image_manifest};
use crate::errors::{RivetResult, invalid_argument};
use arrow::array::{BinaryBuilder, Int32Builder};
use arrow::error::ArrowError;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
use rivet_data::errors::DataError;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type TorchImageSplit = (PathBuf, PathBuf, usize, u32, u32, u8);

pub fn convert_torch_images(
    output: &Path,
    splits: BTreeMap<String, TorchImageSplit>,
    classes: Option<&[String]>,
) -> RivetResult<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(DataError::from)?;
    fs::create_dir(output).map_err(DataError::from)?;
    let schema = native_schema(false);
    let mut rows_by_split = BTreeMap::new();
    for (split, (images, labels, rows, width, height, channels)) in splits {
        validate_split_name(&split)?;
        let color = match channels {
            1 => ExtendedColorType::L8,
            3 => ExtendedColorType::Rgb8,
            _ => {
                return Err(invalid_argument(
                    "torchvision images require 1 or 3 channels",
                ));
            }
        };
        let mut pixels = vec![0; width as usize * height as usize * channels as usize];
        let mut images = BufReader::new(File::open(images).map_err(DataError::from)?);
        let mut labels = BufReader::new(File::open(labels).map_err(DataError::from)?);
        let batch_schema = schema.clone();
        let batches = (0..rows).step_by(512).map(move |start| {
            let mut encoded = BinaryBuilder::new();
            let mut targets = Int32Builder::new();
            let mut png = Vec::new();
            for _ in start..(start + 512).min(rows) {
                images.read_exact(&mut pixels).map_err(ArrowError::from)?;
                let mut label = [0; 4];
                labels.read_exact(&mut label).map_err(ArrowError::from)?;
                png.clear();
                PngEncoder::new(&mut png)
                    .write_image(&pixels, width, height, color)
                    .map_err(|err| ArrowError::ExternalError(Box::new(err)))?;
                encoded.append_value(&png);
                targets.append_value(i32::from_le_bytes(label));
            }
            RecordBatch::try_new(
                batch_schema.clone(),
                vec![Arc::new(encoded.finish()), Arc::new(targets.finish())],
            )
        });
        let path = output.join(format!("{split}.lance"));
        let uri = path
            .to_str()
            .ok_or_else(|| invalid_argument("Lance output path must be UTF-8"))?;
        let reader = RecordBatchIterator::new(batches, schema.clone());
        runtime
            .block_on(lance::Dataset::write(reader, uri, None))
            .map_err(DataError::from)?;
        rows_by_split.insert(split, rows);
    }
    write_image_manifest(output, &rows_by_split, false, "torchvision", classes)
}
