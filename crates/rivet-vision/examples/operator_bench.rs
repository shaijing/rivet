//! Fixed operator-decomposition benchmark for the CIFAR-10 image path.
//!
//! The benchmark intentionally uses the same input shape and statistics as
//! the CIFAR-10 pipeline examples:
//!
//! ```text
//! RIVET_OPERATOR_BENCH_ITERS=20 \
//!   cargo run -p rivet-vision --release --example operator_bench
//! ```
//!
//! The Arrow file can be selected with `RIVET_OPERATOR_BENCH_ARROW` or the
//! test-compatible `RIVET_TEST_ARROW_FILE` variable. If neither is set, the
//! local Hugging Face CIFAR-10 cache is searched.

use rivet_core::Tensor;
use rivet_data::dataset::Dataset;
use rivet_vision::cache::{DecodedImageMemoryDataset, DenseImageMemoryDataset};
use rivet_vision::datasets::ArrowImageDataset;
use rivet_vision::pipeline::ImagePipeline;
use rivet_vision::sample::image::{DecodedSample, ImageLayout, ImageSample};
use rivet_vision::transforms::crop::CropConfig;
use rivet_vision::transforms::decode::decode_rgb;
use rivet_vision::transforms::layout::LayoutConfig;
use rivet_vision::transforms::normalize::{
    normalize_u8_batch_to_f32, normalize_u8_batch_to_nchw_f32, normalize_u8_to_f32,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::error::Error;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use walkdir::WalkDir;

const BATCH_SIZE: usize = 128;
const WORKERS: usize = 4;
const PREFETCH_BATCHES: usize = 2;
const HEIGHT: usize = 32;
const WIDTH: usize = 32;
const CHANNELS: usize = 3;
const IMAGE_BYTES: usize = HEIGHT * WIDTH * CHANNELS;
const BATCH_BYTES: usize = BATCH_SIZE * IMAGE_BYTES;
const MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
const STD: [f32; 3] = [0.2470, 0.2435, 0.2616];

static RECORD_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        record_allocation(pointer, layout.size());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        record_allocation(pointer, layout.size());
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        record_allocation(pointer, new_size);
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

fn record_allocation(pointer: *mut u8, bytes: usize) {
    if RECORD_ALLOCATIONS.load(Ordering::Relaxed) && !pointer.is_null() {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Default)]
struct AllocationStats {
    calls: usize,
    bytes: usize,
}

fn reset_allocation_stats() {
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn allocation_stats() -> AllocationStats {
    AllocationStats {
        calls: ALLOCATION_CALLS.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

struct Measurement {
    elapsed: Duration,
    allocations: AllocationStats,
}

type BenchResult<T> = Result<T, Box<dyn Error>>;

fn measure<F>(iterations: usize, mut operation: F) -> BenchResult<Measurement>
where
    F: FnMut() -> BenchResult<()>,
{
    RECORD_ALLOCATIONS.store(false, Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..iterations {
        operation()?;
    }
    let elapsed = start.elapsed();

    // Allocation counters are sampled separately so their atomic bookkeeping
    // does not distort the reported latency.
    reset_allocation_stats();
    RECORD_ALLOCATIONS.store(true, Ordering::Relaxed);
    operation()?;
    RECORD_ALLOCATIONS.store(false, Ordering::Relaxed);

    Ok(Measurement {
        elapsed,
        allocations: allocation_stats(),
    })
}

fn print_measurement(
    name: &str,
    measurement: &Measurement,
    iterations: usize,
    bytes_copied: usize,
) {
    let seconds = measurement.elapsed.as_secs_f64() / iterations as f64;
    let images_per_second = BATCH_SIZE as f64 / seconds;
    println!(
        "{name:<26} latency_ms={:>10.3} images_per_sec={:>12.0} alloc_calls={:>6} allocated_bytes={:>10} bytes_copied_estimate={bytes_copied}",
        seconds * 1_000.0,
        images_per_second,
        measurement.allocations.calls,
        measurement.allocations.bytes,
    );
}

fn find_arrow_file() -> BenchResult<PathBuf> {
    for variable in ["RIVET_OPERATOR_BENCH_ARROW", "RIVET_TEST_ARROW_FILE"] {
        if let Some(path) = env::var_os(variable) {
            return Ok(PathBuf::from(path));
        }
    }

    let roots = [
        env::var_os("HF_DATASETS_CACHE").map(PathBuf::from),
        env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("datasets")),
        env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/datasets")),
    ];

    for root in roots.into_iter().flatten() {
        if !root.is_dir() {
            continue;
        }
        let mut matches = WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name() == "cifar10-train.arrow")
            .map(|entry| entry.into_path())
            .collect::<Vec<_>>();
        matches.sort();
        if let Some(path) = matches.pop() {
            return Ok(path);
        }
    }

    Err("CIFAR-10 Arrow file not found; set RIVET_OPERATOR_BENCH_ARROW".into())
}

fn load_batch() -> BenchResult<(PathBuf, DecodedImageMemoryDataset, Vec<DecodedSample>)> {
    let path = find_arrow_file()?;
    let dataset = ArrowImageDataset::new(vec![path.clone()], "img".into(), "label".into())?;
    let indices = (0..BATCH_SIZE).collect::<Vec<_>>();
    let encoded = dataset.get_many(&indices)?;
    let decoded = encoded
        .into_iter()
        .map(|sample| decode_rgb(sample.image.as_slice(), sample.label))
        .collect::<Result<Vec<_>, _>>()?;
    let cache = DecodedImageMemoryDataset::from_samples(decoded)?;
    let samples = cache.get_many(&indices)?;

    println!(
        "input: {} | rows={} batch={} shape={}x{}x{}",
        path.display(),
        dataset.len(),
        BATCH_SIZE,
        HEIGHT,
        WIDTH,
        CHANNELS,
    );
    Ok((path, cache, samples))
}

fn dense_from_cache(
    cache: &DecodedImageMemoryDataset,
) -> BenchResult<Arc<DenseImageMemoryDataset>> {
    let dense = cache
        .as_dense()
        .ok_or("CIFAR-10 batch did not produce a dense decoded cache")?;
    Ok(Arc::new(DenseImageMemoryDataset::new(
        dense.images().clone(),
        dense.labels().clone(),
    )?))
}

fn main() -> BenchResult<()> {
    let iterations = env::var("RIVET_OPERATOR_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20usize);
    if iterations == 0 {
        return Err("RIVET_OPERATOR_BENCH_ITERS must be greater than 0".into());
    }

    let (arrow_path, decoded_cache, samples) = load_batch()?;
    let dense = dense_from_cache(&decoded_cache)?;
    let indices = (0..BATCH_SIZE).collect::<Vec<_>>();
    let batch = dense.get_batch(&indices)?;
    let batch_images = batch.images.clone();

    println!(
        "configuration: iterations={iterations} batch={BATCH_SIZE} workers={WORKERS} prefetch={PREFETCH_BATCHES} image_bytes={IMAGE_BYTES} batch_bytes={BATCH_BYTES}"
    );
    println!(
        "benchmark                    latency_ms  images/sec alloc_calls allocated_bytes bytes_copied_estimate"
    );

    let measurement = measure(iterations, || {
        let output = decoded_cache.get_many(&indices)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("decoded_cache.get_many", &measurement, iterations, 0);

    let measurement = measure(iterations, || {
        let output = dense.get_batch(&indices)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("dense_cache.get_batch", &measurement, iterations, 0);

    let crop = CropConfig {
        x: 0,
        y: 0,
        width: WIDTH as u32,
        height: HEIGHT as u32,
    };
    let measurement = measure(iterations, || {
        for sample in &samples {
            let output = crop.apply(ImageSample::Decoded(sample.clone()), ImageLayout::Hwc)?;
            black_box(output);
        }
        Ok(())
    })?;
    print_measurement("crop_view", &measurement, iterations, 0);

    let layout = LayoutConfig {
        layout: ImageLayout::Chw,
    };
    let measurement = measure(iterations, || {
        let output = layout.apply_batch(batch_images.clone(), ImageLayout::Hwc)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("hwc_to_chw_view", &measurement, iterations, 0);

    let contiguous_refs = samples
        .iter()
        .map(|sample| &sample.image)
        .collect::<Vec<_>>();
    let measurement = measure(iterations, || {
        let output = Tensor::stack(&contiguous_refs, 0)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("stack_contiguous", &measurement, iterations, BATCH_BYTES);

    let non_contiguous = samples
        .iter()
        .map(|sample| sample.image.permute(&[1, 0, 2]))
        .collect::<Result<Vec<_>, _>>()?;
    let non_contiguous_refs = non_contiguous.iter().collect::<Vec<_>>();
    let measurement = measure(iterations, || {
        let output = Tensor::stack(&non_contiguous_refs, 0)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement(
        "stack_non_contiguous",
        &measurement,
        iterations,
        BATCH_BYTES,
    );

    let measurement = measure(iterations, || {
        for sample in &samples {
            let output = normalize_u8_to_f32(&sample.image, &MEAN, &STD, ImageLayout::Hwc)?;
            black_box(output);
        }
        Ok(())
    })?;
    print_measurement("sample_normalize", &measurement, iterations, 0);

    let measurement = measure(iterations, || {
        let output = normalize_u8_batch_to_f32(&batch_images, &MEAN, &STD, ImageLayout::Hwc)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("batch_normalize", &measurement, iterations, 0);

    let measurement = measure(iterations, || {
        let normalized = normalize_u8_batch_to_f32(&batch_images, &MEAN, &STD, ImageLayout::Hwc)?;
        let output = normalized.permute(&[0, 3, 1, 2])?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("normalize_plus_layout", &measurement, iterations, 0);

    let measurement = measure(iterations, || {
        let output = normalize_u8_batch_to_nchw_f32(&batch_images, &MEAN, &STD)?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("normalize_to_chw_fused", &measurement, iterations, 0);

    let dataset = Arc::new(ArrowImageDataset::new(
        vec![arrow_path],
        "img".into(),
        "label".into(),
    )?);
    let mut loader = ImagePipeline::new(dataset)
        .decode_image()
        .crop(0, 0, WIDTH as u32, HEIGHT as u32)
        .normalize(MEAN.to_vec(), STD.to_vec())
        .hwc_to_chw()
        .workers(WORKERS)
        .prefetch_batches(PREFETCH_BATCHES)
        // One extra batch is reserved for the post-timing allocation sample.
        .take((iterations + 1) * BATCH_SIZE)
        .batch(BATCH_SIZE, false)
        .compile()?;
    let measurement = measure(iterations, || {
        let output = loader
            .next_batch()?
            .ok_or("full pipeline ended before the benchmark iteration count")?;
        black_box(output);
        Ok(())
    })?;
    print_measurement("full_pipeline", &measurement, iterations, BATCH_BYTES);

    Ok(())
}
