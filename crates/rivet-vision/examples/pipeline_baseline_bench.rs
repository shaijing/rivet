//! Phase-0 pipeline baseline.
//!
//! This is intentionally a simple wall-clock benchmark rather than a
//! statistics-heavy microbenchmark: it records first-batch latency and the
//! steady-state throughput for every worker/prefetch combination that the
//! pipeline plan must preserve.
//!
//! ```text
//! cargo run -j 12 -p rivet-vision --release --example pipeline_baseline_bench
//! ```

use arrow_buffer::Buffer;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use rivet_data::dataset::MemoryDataset;
use rivet_vision::pipeline::ImagePipeline;
use rivet_vision::sample::image::EncodedImageSample;
use std::env;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_SAMPLES: usize = 256;
const DEFAULT_BATCH: usize = 32;
const DEFAULT_BATCHES: usize = 8;

#[derive(Clone, Copy)]
enum Workload {
    Decode,
    RandomCrop,
    RandomCropFlip,
    RandomCropNormalize,
    RandomCropNormalizeChw,
    FullCifar,
    ImageNet,
}

impl Workload {
    const ALL: [Self; 7] = [
        Self::Decode,
        Self::RandomCrop,
        Self::RandomCropFlip,
        Self::RandomCropNormalize,
        Self::RandomCropNormalizeChw,
        Self::FullCifar,
        Self::ImageNet,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::RandomCrop => "random_crop",
            Self::RandomCropFlip => "random_crop_flip",
            Self::RandomCropNormalize => "random_crop_normalize",
            Self::RandomCropNormalizeChw => "random_crop_normalize_chw",
            Self::FullCifar => "full_cifar",
            Self::ImageNet => "imagenet_random_resized_crop_flip_color_jitter_normalize_chw",
        }
    }

    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::ImageNet => (256, 256),
            _ => (32, 32),
        }
    }
}

struct Config {
    samples: usize,
    batch: usize,
    batches: usize,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let value = env::var(name)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    Ok(value.unwrap_or(default))
}

fn encoded_dataset(
    workload: Workload,
    samples: usize,
) -> Result<Arc<MemoryDataset<EncodedImageSample>>, Box<dyn std::error::Error>> {
    let (width, height) = workload.dimensions();
    let image = ImageBuffer::from_fn(width, height, |x, y| {
        Rgb([
            ((x + y) % 251) as u8,
            ((x * 3 + y * 5) % 251) as u8,
            ((x * 7 + y * 11) % 251) as u8,
        ])
    });
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image).write_to(&mut encoded, ImageFormat::Png)?;
    let encoded = Buffer::from(encoded.into_inner());
    let items = (0..samples)
        .map(|index| EncodedImageSample {
            image: encoded.clone(),
            label: index as i64,
        })
        .collect();
    Ok(Arc::new(MemoryDataset::new(items)))
}

fn make_pipeline(
    workload: Workload,
    dataset: Arc<MemoryDataset<EncodedImageSample>>,
    workers: usize,
    prefetch: usize,
) -> ImagePipeline {
    let pipeline = ImagePipeline::new(dataset).decode_image();
    let pipeline = match workload {
        Workload::Decode => pipeline,
        Workload::RandomCrop => pipeline.random_crop(24, 24, 4),
        Workload::RandomCropFlip => pipeline.random_crop(24, 24, 4).random_horizontal_flip(0.5),
        Workload::RandomCropNormalize => pipeline
            .random_crop(24, 24, 4)
            .normalize(vec![0.4914, 0.4822, 0.4465], vec![0.2470, 0.2435, 0.2616]),
        Workload::RandomCropNormalizeChw => pipeline
            .random_crop(24, 24, 4)
            .normalize(vec![0.4914, 0.4822, 0.4465], vec![0.2470, 0.2435, 0.2616])
            .hwc_to_chw(),
        Workload::FullCifar => pipeline
            .random_crop(32, 32, 4)
            .random_horizontal_flip(0.5)
            .normalize(vec![0.4914, 0.4822, 0.4465], vec![0.2470, 0.2435, 0.2616])
            .hwc_to_chw(),
        Workload::ImageNet => pipeline
            .random_resized_crop(224, 224)
            .random_horizontal_flip(0.5)
            .color_jitter(32, 0.4, 12)
            .normalize(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
            .hwc_to_chw(),
    };
    pipeline
        .seed(0xBACE_2026)
        .epoch(0)
        .workers(workers)
        .prefetch_batches(prefetch)
        .batch(DEFAULT_BATCH, false)
}

fn run_once(
    workload: Workload,
    config: &Config,
    workers: usize,
    prefetch: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dataset = encoded_dataset(workload, config.samples)?;
    let mut loader = make_pipeline(workload, dataset, workers, prefetch)
        .batch(config.batch, false)
        .compile()?;

    let first_start = Instant::now();
    let first = loader.next_batch()?.ok_or("benchmark source ended early")?;
    let first_ms = first_start.elapsed().as_secs_f64() * 1_000.0;
    let first_images = first.images.dims()[0];

    let steady_start = Instant::now();
    let mut batches = 1usize;
    let mut images = first_images;
    while batches < config.batches {
        let Some(batch) = loader.next_batch()? else {
            break;
        };
        images += batch.images.dims()[0];
        batches += 1;
    }
    let steady_elapsed = steady_start.elapsed().as_secs_f64();
    let steady_batches = batches.saturating_sub(1);
    let steady_ms = if steady_batches == 0 {
        0.0
    } else {
        steady_elapsed * 1_000.0 / steady_batches as f64
    };
    let images_per_sec = if steady_elapsed == 0.0 {
        0.0
    } else {
        images.saturating_sub(first_images) as f64 / steady_elapsed
    };
    println!(
        "{:<58} workers={:<4} prefetch={:<2} first_ms={:>9.3} steady_ms={:>9.3} img_s={:>10.1}",
        workload.name(),
        workers,
        prefetch,
        first_ms,
        steady_ms,
        images_per_sec
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        samples: env_usize("RIVET_PIPELINE_BENCH_SAMPLES", DEFAULT_SAMPLES)?,
        batch: env_usize("RIVET_PIPELINE_BENCH_BATCH", DEFAULT_BATCH)?,
        batches: env_usize("RIVET_PIPELINE_BENCH_BATCHES", DEFAULT_BATCHES)?,
    };
    if config.samples < config.batch || config.batch == 0 || config.batches < 2 {
        return Err("samples must be >= batch > 0 and batches must be >= 2".into());
    }

    let physical = num_cpus::get_physical();
    let workers = [0, 1, 4, physical];
    let prefetch = [0, 1, 2];
    println!(
        "samples={} batch={} batches={} physical_cores={}",
        config.samples, config.batch, config.batches, physical
    );
    println!("workload                                                     workers prefetch  first_ms steady_ms      img_s");
    for workload in Workload::ALL {
        for &worker_count in &workers {
            for &prefetch_count in &prefetch {
                run_once(workload, &config, worker_count, prefetch_count)?;
            }
        }
    }
    Ok(())
}
