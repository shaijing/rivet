//! Compare revisions with identical BLAS/Rayon thread counts. Allocation
//! counts include Rust allocations only, not allocations inside native BLAS.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rivet_core::{DType, Device, Tensor};
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
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, len: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, len) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_matmul");
    for (count, size) in [(512, 8), (128, 32), (4, 256)] {
        for layout in ["dense", "broadcast", "transpose", "strided", "irregular"] {
            let a = Tensor::ones((count, size, size), DType::F32, &Device::Cpu).unwrap();
            let b = match layout {
                "broadcast" => Tensor::ones((1, size, size), DType::F32, &Device::Cpu).unwrap(),
                "transpose" => a.transpose(1, 2).unwrap(),
                "strided" => Tensor::ones((count, size, size * 2), DType::F32, &Device::Cpu)
                    .unwrap()
                    .narrow(2, 0, size)
                    .unwrap(),
                // Interleaved batches cannot be flattened to a constant stride.
                "irregular" => Tensor::ones((count / 2, 2, size, size), DType::F32, &Device::Cpu)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap(),
                _ => a.clone(),
            };
            let a = if layout == "irregular" {
                a.reshape((2, count / 2, size, size)).unwrap()
            } else {
                a
            };
            let add = Tensor::ones(a.dims().to_vec(), DType::F32, &Device::Cpu).unwrap();
            let reduce = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
            for op in ["matmul", "baddbmm", "addbmm"] {
                let run = || match op {
                    "matmul" => a.broadcast_matmul(&b).unwrap(),
                    "baddbmm" => add.baddbmm(&a, &b, 0.5, 2.0).unwrap(),
                    _ => reduce.addbmm(&a, &b, 0.5, 2.0).unwrap(),
                };
                let result = run();
                let expected = match op {
                    "matmul" => size as f32,
                    "baddbmm" => size as f32 * 0.5 + 2.0,
                    _ => (count * size) as f32 * 0.5 + 2.0,
                };
                assert!(
                    result
                        .to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| *v == expected)
                );
                ALLOCATIONS.store(0, Ordering::Relaxed);
                COUNTING.store(true, Ordering::Relaxed);
                black_box(run());
                COUNTING.store(false, Ordering::Relaxed);
                let id = format!("{op}/{layout}/{count}x{size}");
                eprintln!(
                    "Rust allocations {id}: {}",
                    ALLOCATIONS.load(Ordering::Relaxed)
                );
                group.bench_function(BenchmarkId::from_parameter(id), |bench| {
                    bench.iter(|| black_box(run()));
                });
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250))
        .sample_size(10);
    targets = batches
}
criterion_main!(benches);
