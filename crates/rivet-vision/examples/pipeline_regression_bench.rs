use rivet_core::{Device, Tensor};
use rivet_vision::{DecodedSample, ImagePipeline, ImageSource};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const BATCH: usize = 128;
const ITERATIONS: usize = 40;
const SAMPLES: usize = (ITERATIONS + 2) * BATCH;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Tensor::from_vec(vec![127u8; 32 * 32 * 3], [32, 32, 3], &Device::Cpu)?;
    let samples = (0..SAMPLES)
        .map(|index| DecodedSample {
            image: image.clone(),
            label: index as i64,
        })
        .collect();
    let source = ImageSource::from_decoded(Arc::new(rivet_vision::DecodedImageMemoryDataset::new(
        samples,
    )));
    let mut loader = ImagePipeline::from_source(source)
        .resize(32, 32)
        .normalize(vec![0.5; 3], vec![0.5; 3])
        .hwc_to_chw()
        .workers(4)
        .prefetch_batches(2)
        .batch(BATCH, false)
        .compile()?;

    println!(
        "input_state={:?} pre_batch_state={:?} output_state={:?} sample_ops={} batch_ops={} first_sample={} first_batch={}",
        loader.plan.input_state,
        loader.plan.pre_batch_state,
        loader.plan.output_state,
        loader.plan.sample_op_count(),
        loader.plan.batch_op_count(),
        loader.plan.first_sample_op_name().unwrap_or("-"),
        loader.plan.first_batch_op_name().unwrap_or("-"),
    );
    let first_start = Instant::now();
    black_box(loader.next_batch()?.ok_or("warmup ended")?);
    let first_batch_ms = first_start.elapsed().as_secs_f64() * 1_000.0;

    let mut batch_latencies_ms = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        black_box(loader.next_batch()?.ok_or("benchmark ended")?);
        batch_latencies_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let elapsed = batch_latencies_ms.iter().sum::<f64>() / 1_000.0;
    let mut sorted_latencies_ms = batch_latencies_ms.clone();
    sorted_latencies_ms.sort_by(f64::total_cmp);
    let percentile = |quantile: f64| {
        let rank = (quantile * sorted_latencies_ms.len() as f64).ceil() as usize;
        sorted_latencies_ms[rank.saturating_sub(1).min(sorted_latencies_ms.len() - 1)]
    };
    println!(
        "first_batch_ms={first_batch_ms:.3} steady_batch_p50_ms={:.3} steady_batch_p95_ms={:.3} seconds={elapsed:.6} images_per_sec={:.0}",
        percentile(0.50),
        percentile(0.95),
        ITERATIONS as f64 * BATCH as f64 / elapsed,
    );
    Ok(())
}
