//! Run with matched Rayon and native BLAS thread counts, with or without
//! `--features blas`, to compare fused operations and the old batch strategy.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rivet_core::{DType, Device, Tensor};
use std::hint::black_box;
use std::time::Duration;

fn check(lhs: &Tensor, rhs: &Tensor) {
    assert_eq!(lhs.dims(), rhs.dims());
    for (left, right) in lhs
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(rhs.to_vec::<f32>().unwrap())
    {
        assert!((left - right).abs() <= 1e-4 * (1.0 + right.abs()));
    }
}

fn fused(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("linalg_fused");
    for size in [32, 256] {
        let a = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
        let c = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
        check(
            &c.addmm(&a, &b, 1.0, 1.0).unwrap(),
            &a.matmul(&b).unwrap().add(&c).unwrap(),
        );
        group.bench_function(BenchmarkId::new("matmul_add", size), |bench| {
            bench.iter(|| black_box(a.matmul(&b).unwrap().add(&c).unwrap()));
        });
        group.bench_function(BenchmarkId::new("addmm", size), |bench| {
            bench.iter(|| black_box(c.addmm(&a, &b, 1.0, 1.0).unwrap()));
        });
        let vector = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
        check(
            &vector.addmv(&a, &vector, 1.0, 1.0).unwrap(),
            &a.mv(&vector).unwrap().add(&vector).unwrap(),
        );
        group.bench_function(BenchmarkId::new("mv_add", size), |bench| {
            bench.iter(|| black_box(a.mv(&vector).unwrap().add(&vector).unwrap()));
        });
        group.bench_function(BenchmarkId::new("addmv", size), |bench| {
            bench.iter(|| black_box(vector.addmv(&a, &vector, 1.0, 1.0).unwrap()));
        });
        check(
            &a.gram(true).unwrap(),
            &a.transpose(0, 1).unwrap().matmul(&a).unwrap(),
        );
        group.bench_function(BenchmarkId::new("transpose_matmul", size), |bench| {
            bench.iter(|| black_box(a.transpose(0, 1).unwrap().matmul(&a).unwrap()));
        });
        group.bench_function(BenchmarkId::new("gram", size), |bench| {
            bench.iter(|| black_box(a.gram(true).unwrap()));
        });
    }
    group.finish();
}

// The pre-optimization broadcast/contiguous, per-batch matmul, and stack path.
fn legacy_batch(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let rhs = rhs
        .broadcast_as((lhs.dims()[0], rhs.dims()[1], rhs.dims()[2]))
        .unwrap()
        .contiguous()
        .unwrap();
    let lhs = lhs.contiguous().unwrap();
    let products = (0..lhs.dims()[0])
        .map(|batch| {
            lhs.get(batch)
                .unwrap()
                .matmul(&rhs.get(batch).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    Tensor::stack(&products.iter().collect::<Vec<_>>(), 0).unwrap()
}

fn batches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("linalg_batch");
    for (count, size) in [(128, 32), (32, 128)] {
        let lhs = Tensor::ones((count, size, size), DType::F32, &Device::Cpu).unwrap();
        let rhs = Tensor::ones((1, size, size), DType::F32, &Device::Cpu).unwrap();
        check(
            &lhs.broadcast_matmul(&rhs).unwrap(),
            &legacy_batch(&lhs, &rhs),
        );
        let shape = format!("{count}x{size}x{size}");
        group.bench_function(BenchmarkId::new("legacy", &shape), |bench| {
            bench.iter(|| black_box(legacy_batch(&lhs, &rhs)));
        });
        group.bench_function(BenchmarkId::new("direct", &shape), |bench| {
            bench.iter(|| black_box(lhs.broadcast_matmul(&rhs).unwrap()));
        });
    }
    group.finish();
}

fn extensions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("linalg_extensions");
    let (count, size) = (32, 64);
    let a = Tensor::ones((count, size, size), DType::F32, &Device::Cpu).unwrap();
    let b = Tensor::ones((1, size, size), DType::F32, &Device::Cpu).unwrap();
    let c = Tensor::ones((count, size, size), DType::F32, &Device::Cpu).unwrap();
    check(
        &c.baddbmm(&a, &b, 1.0, 1.0).unwrap(),
        &a.broadcast_matmul(&b).unwrap().add(&c).unwrap(),
    );
    group.bench_function("batch_matmul_add", |bench| {
        bench.iter(|| black_box(a.broadcast_matmul(&b).unwrap().add(&c).unwrap()))
    });
    group.bench_function("baddbmm", |bench| {
        bench.iter(|| black_box(c.baddbmm(&a, &b, 1.0, 1.0).unwrap()))
    });
    let c = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
    check(
        &c.addbmm(&a, &b, 1.0, 1.0).unwrap(),
        &a.broadcast_matmul(&b)
            .unwrap()
            .sum(0)
            .unwrap()
            .add(&c)
            .unwrap(),
    );
    group.bench_function("batch_matmul_sum_add", |bench| {
        bench.iter(|| {
            black_box(
                a.broadcast_matmul(&b)
                    .unwrap()
                    .sum(0)
                    .unwrap()
                    .add(&c)
                    .unwrap(),
            )
        })
    });
    group.bench_function("addbmm", |bench| {
        bench.iter(|| black_box(c.addbmm(&a, &b, 1.0, 1.0).unwrap()))
    });
    let a = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
    let b = Tensor::ones((size, size), DType::F32, &Device::Cpu).unwrap();
    group.bench_function("addmm_allocated", |bench| {
        bench.iter(|| black_box(c.addmm(&a, &b, 1.0, 0.0).unwrap()))
    });
    group.bench_function("addmm_exclusive", |bench| {
        bench.iter_batched(
            || {
                Tensor::ones((size, size), DType::F32, &Device::Cpu)
                    .unwrap()
                    .try_into_exclusive()
                    .unwrap()
            },
            |mut dst| {
                dst.addmm(&a, &b, 1.0, 0.0).unwrap();
                black_box(dst)
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(500))
        .sample_size(20);
    targets = fused, batches, extensions
}
criterion_main!(benches);
