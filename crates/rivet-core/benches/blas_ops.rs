//! Compare with identical native thread settings:
//! `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench -j 12 -p rivet-core --bench blas_ops -- --save-baseline default`
//! `FLEXIBLAS=OPENBLAS-SERIAL OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench -j 12 -p rivet-core --bench blas_ops --features blas -- --baseline default`
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rivet_core::{DType, Device, Tensor};
use std::hint::black_box;
use std::time::Duration;

fn vector_reductions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("vector_reductions");
    for len in [32, 4096, 262144] {
        for (name, dtype) in [("f32", DType::F32), ("f64", DType::F64)] {
            let lhs = Tensor::ones(len, dtype, &Device::Cpu).unwrap();
            let rhs = Tensor::ones(len, dtype, &Device::Cpu).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("dot_{name}"), len),
                &len,
                |bench, _| bench.iter(|| black_box(black_box(&lhs).dot(black_box(&rhs)).unwrap())),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("norm_{name}"), len),
                &len,
                |bench, _| bench.iter(|| black_box(black_box(&lhs).norm().unwrap())),
            );
        }
    }
    group.finish();
}

fn matrix_vector(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("matrix_vector");
    for size in [32, 512, 2048] {
        let matrix = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
        let vector = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| black_box(black_box(&matrix).mv(black_box(&vector)).unwrap()));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(20);
    targets = vector_reductions, matrix_vector
}
criterion_main!(benches);
