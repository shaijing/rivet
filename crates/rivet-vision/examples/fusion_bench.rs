//! Compare unfused batch normalization plus layout conversion with the
//! conservative fused NHWC U8 -> NCHW F32 path.
//!
//! ```text
//! cargo run -p rivet-vision --release --example fusion_bench
//! RIVET_FUSION_BENCH_ITERS=20 cargo run -p rivet-vision --release --example fusion_bench
//! ```

use rivet_core::{Device, Tensor};
use rivet_vision::sample::image::ImageAxisOrder;
use rivet_vision::transforms::representation::{
    normalize_u8_batch_to_f32, normalize_u8_batch_to_nchw_f32,
};
use std::env;
use std::hint::black_box;
use std::time::Instant;

const BATCH_SIZE: usize = 128;
const HEIGHT: usize = 32;
const WIDTH: usize = 32;
const CHANNELS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = env::var("RIVET_FUSION_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10usize);
    if iterations == 0 {
        return Err("RIVET_FUSION_BENCH_ITERS must be greater than 0".into());
    }

    let image_size = HEIGHT * WIDTH * CHANNELS;
    let values = (0..BATCH_SIZE * image_size)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let batch = Tensor::from_vec(values, [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS], &Device::Cpu)?;
    let mean = [0.4914, 0.4822, 0.4465];
    let std = [0.2470, 0.2435, 0.2616];

    let start = Instant::now();
    let mut unfused_bytes = 0usize;
    for _ in 0..iterations {
        let normalized = normalize_u8_batch_to_f32(&batch, &mean, &std, ImageAxisOrder::Hwc)?;
        let output = normalized.permute(&[0, 3, 1, 2])?;
        unfused_bytes = unfused_bytes.wrapping_add(output.storage_bytes());
        black_box(&output);
    }
    let unfused_elapsed = start.elapsed();

    let start = Instant::now();
    let mut fused_bytes = 0usize;
    for _ in 0..iterations {
        let output = normalize_u8_batch_to_nchw_f32(&batch, &mean, &std)?;
        fused_bytes = fused_bytes.wrapping_add(output.storage_bytes());
        black_box(&output);
    }
    let fused_elapsed = start.elapsed();

    println!(
        "configuration: batch={BATCH_SIZE} shape={HEIGHT}x{WIDTH}x{CHANNELS} iterations={iterations}"
    );
    println!(
        "unfused     : {:?} ({unfused_bytes} output bytes)",
        unfused_elapsed
    );
    println!(
        "fused       : {:?} ({fused_bytes} output bytes)",
        fused_elapsed
    );
    if fused_elapsed.as_nanos() > 0 {
        println!(
            "speedup     : {:.2}x",
            unfused_elapsed.as_secs_f64() / fused_elapsed.as_secs_f64()
        );
    }

    Ok(())
}
