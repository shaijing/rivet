//! Fix provider/Rayon thread counts. Each case compares the legacy loop+stack
//! with one allocated batch output and, for solves, reused RHS storage.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rivet_core::{DType, Device, MatrixSide, Tensor, TriangularOptions};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

struct Counter;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

fn count<F: FnOnce()>(run: F) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    run();
    COUNTING.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}
fn same(a: &Tensor, b: &Tensor) {
    assert_eq!(a.dims(), b.dims());
    for (a, b) in a
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(b.to_vec::<f32>().unwrap())
    {
        assert!((*a - b).abs() <= 1e-4 * (1. + b.abs()));
    }
}
fn legacy_mv(a: &Tensor, x: &Tensor) -> Tensor {
    let out = (0..a.dims()[0])
        .map(|i| a.get(i).unwrap().mv(&x.get(i).unwrap()).unwrap())
        .collect::<Vec<_>>();
    Tensor::stack(&out.iter().collect::<Vec<_>>(), 0).unwrap()
}
fn legacy_solve(a: &Tensor, b: &Tensor, o: TriangularOptions) -> Tensor {
    let out = (0..b.dims()[0])
        .map(|i| {
            a.get(i)
                .unwrap()
                .triangular_solve_with_options(&b.get(i).unwrap(), o)
                .unwrap()
        })
        .collect::<Vec<_>>();
    Tensor::stack(&out.iter().collect::<Vec<_>>(), 0).unwrap()
}
fn vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_mv");
    for (b, n) in [(512, 8), (8, 128)] {
        for layout in ["dense", "strided"] {
            let a = if layout == "dense" {
                Tensor::ones((b, n, n), DType::F32, &Device::Cpu).unwrap()
            } else {
                Tensor::ones((b, n, n, 2), DType::F32, &Device::Cpu)
                    .unwrap()
                    .get_on_dim(3, 0)
                    .unwrap()
            };
            let x = Tensor::ones((b, n), DType::F32, &Device::Cpu).unwrap();
            same(&a.batch_mv(&x, false).unwrap(), &legacy_mv(&a, &x));
            let id = format!("{layout}/{b}x{n}");
            eprintln!(
                "Rust allocations mv/{id} legacy: {}",
                count(|| {
                    black_box(legacy_mv(&a, &x));
                })
            );
            eprintln!(
                "Rust allocations mv/{id} batch: {}",
                count(|| {
                    black_box(a.batch_mv(&x, false).unwrap());
                })
            );
            group.bench_function(BenchmarkId::new("legacy", &id), |bench| {
                bench.iter(|| black_box(legacy_mv(&a, &x)))
            });
            group.bench_function(BenchmarkId::new("batch", &id), |bench| {
                bench.iter(|| black_box(a.batch_mv(&x, false).unwrap()))
            });
        }
    }
    group.finish();
}
fn solves(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_triangular");
    for (b, n) in [(512, 8), (8, 128)] {
        for shared in [false, true] {
            for side in [MatrixSide::Left, MatrixSide::Right] {
                for nrhs in [1, 4] {
                    let stored = if shared { 1 } else { b };
                    let mut data = vec![f32::NAN; stored * n * n];
                    for batch in 0..stored {
                        for row in 0..n {
                            for col in 0..=row {
                                data[(batch * n + row) * n + col] =
                                    if row == col { 2. } else { 0.01 };
                            }
                        }
                    }
                    let a = Tensor::from_vec(data, (stored, n, n), &Device::Cpu)
                        .unwrap()
                        .broadcast_as((b, n, n))
                        .unwrap();
                    let (rows, cols) = if side == MatrixSide::Left {
                        (n, nrhs)
                    } else {
                        (nrhs, n)
                    };
                    let rhs = Tensor::ones((b, rows, cols), DType::F32, &Device::Cpu).unwrap();
                    let o = TriangularOptions {
                        side,
                        ..Default::default()
                    };
                    let out = a.batch_triangular_solve(&rhs, o).unwrap();
                    same(&out, &legacy_solve(&a, &rhs, o));
                    let mut owned = Tensor::ones(rhs.shape().clone(), DType::F32, &Device::Cpu)
                        .unwrap()
                        .try_into_exclusive()
                        .unwrap();
                    owned.batch_triangular_solve(&a, o).unwrap();
                    let checked = owned.into_tensor();
                    same(&checked, &out);
                    let mut owned = checked.try_into_exclusive().unwrap();
                    let id = format!(
                        "{side:?}/{}/{b}x{n}/rhs{nrhs}",
                        if shared { "shared" } else { "varying" }
                    );
                    eprintln!(
                        "Rust allocations solve/{id} legacy: {}",
                        count(|| {
                            black_box(legacy_solve(&a, &rhs, o));
                        })
                    );
                    eprintln!(
                        "Rust allocations solve/{id} batch: {}",
                        count(|| {
                            black_box(a.batch_triangular_solve(&rhs, o).unwrap());
                        })
                    );
                    eprintln!(
                        "Rust allocations solve/{id} reuse: {}",
                        count(|| {
                            owned.copy_from(&rhs).unwrap();
                            owned.batch_triangular_solve(&a, o).unwrap();
                        })
                    );
                    group.bench_function(BenchmarkId::new("legacy", &id), |bench| {
                        bench.iter(|| black_box(legacy_solve(&a, &rhs, o)))
                    });
                    group.bench_function(BenchmarkId::new("batch", &id), |bench| {
                        bench.iter(|| black_box(a.batch_triangular_solve(&rhs, o).unwrap()))
                    });
                    // Reset RHS inside the timed loop: every solve receives identical data,
                    // and copy cost is included just as for the immutable allocated output.
                    group.bench_function(BenchmarkId::new("reuse_with_rhs_reset", &id), |bench| {
                        bench.iter(|| {
                            owned.copy_from(&rhs).unwrap();
                            owned.batch_triangular_solve(&a, o).unwrap();
                            black_box(&owned);
                        })
                    });
                }
            }
        }
    }
    group.finish();
}
criterion_group! {
    name=benches;
    config=Criterion::default().warm_up_time(Duration::from_millis(50))
        .measurement_time(Duration::from_millis(200)).sample_size(10).nresamples(10_000);
    targets=vectors,solves
}
criterion_main!(benches);
