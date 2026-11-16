//! Compare allocated outputs with a destination reused across iterations.
//! Fix BLAS/Rayon thread counts between runs. Allocation counts are Rust-only.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rivet_core::{DType, Device, ExclusiveTensor, Tensor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, len: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, len) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count<F: FnOnce()>(run: F) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    run();
    COUNTING.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

fn allocated(op: &str, c: &Tensor, a: &Tensor, b: &Tensor, x: &Tensor) -> Tensor {
    match op {
        "baddbmm" => c.baddbmm(a, b, 1.0, 0.0).unwrap(),
        "addbmm" => c.addbmm(a, b, 1.0, 0.0).unwrap(),
        "gram" => a.gram(false).unwrap(),
        "gram_update" => c.gram_update(a, false, 1.0, 0.0, true).unwrap(),
        "rank1" => c.symmetric_rank1_update(x, 1.0, 0.0, true).unwrap(),
        "rank2" => c.symmetric_rank2_update(x, x, 1.0, 0.0, true).unwrap(),
        _ => c
            .symmetric_rank2k_update(a, b, false, 1.0, 0.0, true)
            .unwrap(),
    }
}

fn inplace(op: &str, c: &mut ExclusiveTensor, a: &Tensor, b: &Tensor, x: &Tensor) {
    match op {
        "baddbmm" => c.baddbmm(a, b, 1.0, 0.0).unwrap(),
        "addbmm" => c.addbmm(a, b, 1.0, 0.0).unwrap(),
        "gram" => c.gram(a, false).unwrap(),
        "gram_update" => c.gram_update(a, false, 1.0, 0.0, true).unwrap(),
        "rank1" => c.symmetric_rank1_update(x, 1.0, 0.0, true).unwrap(),
        "rank2" => c.symmetric_rank2_update(x, x, 1.0, 0.0, true).unwrap(),
        _ => c
            .symmetric_rank2k_update(a, b, false, 1.0, 0.0, true)
            .unwrap(),
    }
}

fn outputs(c: &mut Criterion) {
    let mut group = c.benchmark_group("exclusive_linalg");
    for size in [32, 256] {
        let batches = if size == 32 { 128 } else { 4 };
        let matrix = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
        let batch = Tensor::ones((batches, size, size), DType::F32, &Device::Cpu).unwrap();
        let shared = matrix.unsqueeze(0).unwrap();
        for op in [
            "baddbmm",
            "addbmm",
            "gram",
            "gram_update",
            "rank1",
            "rank2",
            "rank2k",
        ] {
            let (a, b) = if op == "baddbmm" || op == "addbmm" {
                (&batch, &shared)
            } else {
                (&matrix, &matrix)
            };
            let shape = if op == "baddbmm" {
                batch.shape()
            } else {
                matrix.shape()
            };
            let seed = Tensor::ones(shape.clone(), DType::F32, &Device::Cpu).unwrap();
            let mut output = Tensor::ones(shape.clone(), DType::F32, &Device::Cpu)
                .unwrap()
                .try_into_exclusive()
                .unwrap();
            inplace(op, &mut output, a, b, &x);
            let expected = allocated(op, &seed, a, b, &x);
            let checked = output.into_tensor();
            assert_eq!(
                checked.to_vec::<f32>().unwrap(),
                expected.to_vec::<f32>().unwrap()
            );
            let mut output = checked.try_into_exclusive().unwrap();
            let id = format!("{op}/{size}");
            eprintln!(
                "Rust allocations {id} allocated: {}",
                count(|| {
                    black_box(allocated(op, &seed, a, b, &x));
                })
            );
            eprintln!(
                "Rust allocations {id} inplace: {}",
                count(|| {
                    inplace(op, &mut output, a, b, &x);
                })
            );
            group.bench_function(BenchmarkId::new("allocated", &id), |bench| {
                bench.iter(|| black_box(allocated(op, &seed, a, b, &x)));
            });
            group.bench_function(BenchmarkId::new("inplace", &id), |bench| {
                bench.iter(|| {
                    inplace(op, &mut output, a, b, &x);
                    black_box(&output);
                });
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250)).sample_size(10);
    targets = outputs
}
criterion_main!(benches);
