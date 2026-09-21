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
    let source = ImageSource::from_decoded(Arc::new(rivet_vision::DecodedImageMemoryDataset::new(samples)));
    let mut loader = ImagePipeline::from_source(source)
        .resize(32, 32)
        .normalize(vec![0.5; 3], vec![0.5; 3])
        .hwc_to_chw()
        .workers(4)
        .prefetch_batches(2)
        .batch(BATCH, false)
        .compile()?;

    println!(
        "sample_ops={} batch_ops={} first_sample={} first_batch={}",
        loader.plan.sample_ops.len(),
        loader.plan.batch_ops.len(),
        loader.plan.sample_ops.first().map(|op| op.name()).unwrap_or("-"),
        loader.plan.batch_ops.first().map(|op| op.name()).unwrap_or("-"),
    );
    black_box(loader.next_batch()?.ok_or("warmup ended")?);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(loader.next_batch()?.ok_or("benchmark ended")?);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("seconds={elapsed:.6} images_per_sec={:.0}", ITERATIONS as f64 * BATCH as f64 / elapsed);
    Ok(())
}
