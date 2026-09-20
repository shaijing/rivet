//! Compare per-sample and batch-stage image dtype conversion.
//!
//! ```text
//! RIVET_DTYPE_BENCH_ITERS=20 cargo run -p rivet-vision --release --example dtype_bench
//! ```

use rivet_core::{DType, Device, Tensor};
use rivet_vision::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_vision::transforms::representation::ConvertImageDtypeConfig;
use std::env;
use std::hint::black_box;
use std::time::Instant;

const BATCH_SIZE: usize = 128;
const HEIGHT: usize = 32;
const WIDTH: usize = 32;
const CHANNELS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = env::var("RIVET_DTYPE_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10usize);
    if iterations == 0 {
        return Err("RIVET_DTYPE_BENCH_ITERS must be greater than 0".into());
    }

    let image_size = HEIGHT * WIDTH * CHANNELS;
    let batch = Tensor::from_vec(
        (0..BATCH_SIZE * image_size)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>(),
        [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS],
        &Device::Cpu,
    )?;
    let samples = (0..BATCH_SIZE)
        .map(|index| batch.narrow(0, index, 1)?.squeeze(0))
        .collect::<Result<Vec<_>, _>>()?;
    let config = ConvertImageDtypeConfig::new(DType::F32);

    let start = Instant::now();
    let mut sample_bytes = 0usize;
    for _ in 0..iterations {
        for sample in &samples {
            let output = config.apply(
                ImageSample::Decoded(DecodedSample {
                    image: sample.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )?;
            let output = output.into_decoded()?.image;
            sample_bytes = sample_bytes.wrapping_add(output.storage_bytes());
            black_box(&output);
        }
    }
    let sample_elapsed = start.elapsed();

    let start = Instant::now();
    let mut batch_bytes = 0usize;
    for _ in 0..iterations {
        let output = config.apply_batch(batch.clone(), ImageAxisOrder::Hwc)?;
        batch_bytes = batch_bytes.wrapping_add(output.storage_bytes());
        black_box(&batch_bytes);
    }
    let batch_elapsed = start.elapsed();

    println!(
        "configuration: batch={BATCH_SIZE} shape={HEIGHT}x{WIDTH}x{CHANNELS} iterations={iterations}"
    );
    println!(
        "sample-stage: {:?} ({sample_bytes} output bytes)",
        sample_elapsed
    );
    println!(
        "batch-stage : {:?} ({batch_bytes} output bytes)",
        batch_elapsed
    );
    if batch_elapsed.as_nanos() > 0 {
        println!(
            "speedup     : {:.2}x",
            sample_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64()
        );
    }

    Ok(())
}
