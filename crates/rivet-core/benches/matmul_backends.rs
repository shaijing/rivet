//! Run both actual Rivet kernels in one process with identical aligned inputs
//! and output allocation. Fix both libraries' thread counts for a fair run:
//! `RAYON_NUM_THREADS=1 FLEXIBLAS=OPENBLAS-SERIAL OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench -j 12 -p rivet-core --bench matmul_backends --features blas,bench-internals`
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rivet_core::bench::MatmulCase;
use std::hint::black_box;
use std::time::Duration;

fn matmul_backends(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("matmul_backends");
    for (m, k, n) in [
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (64, 1024, 256),
        (1024, 64, 256),
        (512, 512, 1),
    ] {
        let case = MatmulCase::new(m, k, n).unwrap();
        // Check every result before timing so performance numbers cannot hide
        // a backend mismatch. These dyadic inputs keep accumulation exact for
        // the benchmark sizes, but tolerate normal backend rounding changes.
        let gemm = case.gemm().unwrap();
        let blas = case.blas().unwrap();
        assert_eq!(gemm.as_ref().len(), blas.as_ref().len());
        for (&reference, &native) in gemm.as_ref().iter().zip(blas.as_ref()) {
            assert!(
                (reference - native).abs() <= 1e-4 + 1e-4 * reference.abs(),
                "backend mismatch at {m}x{k}x{n}: gemm={reference}, blas={native}"
            );
        }
        drop((gemm, blas));

        let shape = format!("{m}x{k}x{n}");
        // Criterion reports conventional 2*m*k*n floating-point operations.
        group.throughput(Throughput::Elements(2 * m as u64 * k as u64 * n as u64));
        group.bench_function(BenchmarkId::new("gemm", &shape), |bench| {
            bench.iter(|| black_box(black_box(&case).gemm().unwrap()));
        });
        group.bench_function(BenchmarkId::new("blas", &shape), |bench| {
            bench.iter(|| black_box(black_box(&case).blas().unwrap()));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1))
        .sample_size(20);
    targets = matmul_backends
}
criterion_main!(benches);
