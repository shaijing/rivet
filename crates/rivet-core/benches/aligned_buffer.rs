//! Run with:
//! `cargo bench -j 12 -p rivet-core --bench aligned_buffer --features bench-internals`
//!
//! The five data sizes cover small image metadata through multi-megabyte batch
//! outputs.  Access benchmarks use a stride of two so that the strided paths
//! measure real non-contiguous traversal; the contiguous case measures the
//! dedicated slice fast path over the same logical byte count.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rivet_core::bench::{
    builder_extend_from_slice, builder_sequential, sum_contiguous, sum_per_element_get,
    sum_validated_strided,
};
use rivet_core::{Layout, Shape};
use std::hint::black_box;

const SIZES: [(&str, usize); 5] = [
    ("3 KiB", 3 * 1024),
    ("12 KiB", 12 * 1024),
    ("384 KiB", 384 * 1024),
    ("1.5 MiB", 1536 * 1024),
    ("12 MiB", 12 * 1024 * 1024),
];

fn input(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 251) as u8).collect()
}

fn bench_construction(criterion: &mut Criterion) {
    let mut sequential = criterion.benchmark_group("construction/sequential");

    for (label, len) in SIZES {
        let values = input(len);
        sequential.throughput(Throughput::Bytes(len as u64));
        sequential.bench_with_input(BenchmarkId::new("Vec", label), &values, |bench, values| {
            bench.iter(|| {
                let mut output = Vec::with_capacity(values.len());
                for &value in black_box(values) {
                    output.push(value);
                }
                black_box(output)
            });
        });
        sequential.bench_with_input(
            BenchmarkId::new("AlignedBufferBuilder", label),
            &values,
            |bench, values| {
                bench.iter(|| black_box(builder_sequential(black_box(values)).unwrap()));
            },
        );
    }
    sequential.finish();

    let mut bulk = criterion.benchmark_group("construction/extend_from_slice");
    for (label, len) in SIZES {
        let values = input(len);
        bulk.throughput(Throughput::Bytes(len as u64));
        bulk.bench_with_input(BenchmarkId::new("Vec", label), &values, |bench, values| {
            bench.iter(|| {
                let mut output = Vec::with_capacity(values.len());
                output.extend_from_slice(black_box(values));
                black_box(output)
            });
        });
        bulk.bench_with_input(
            BenchmarkId::new("AlignedBufferBuilder", label),
            &values,
            |bench, values| {
                bench.iter(|| black_box(builder_extend_from_slice(black_box(values)).unwrap()));
            },
        );
    }
    bulk.finish();
}

fn bench_access(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("access");
    for (label, len) in SIZES {
        let contiguous = input(len);
        let mut strided = vec![0u8; len * 2];
        for (index, &value) in contiguous.iter().enumerate() {
            strided[index * 2] = value;
        }
        let layout = Layout::new(Shape::from([len, 1]), vec![2, 1], 0).unwrap();

        group.throughput(Throughput::Bytes(len as u64));
        group.bench_with_input(
            BenchmarkId::new("per_element_get", label),
            &(&strided, &layout),
            |bench, (values, layout)| {
                bench.iter(|| {
                    black_box(sum_per_element_get(black_box(values), black_box(layout)).unwrap())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("validated_strided", label),
            &(&strided, &layout),
            |bench, (values, layout)| {
                bench.iter(|| {
                    black_box(sum_validated_strided(black_box(values), black_box(layout)).unwrap())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("contiguous_slice", label),
            &contiguous,
            |bench, values| bench.iter(|| black_box(sum_contiguous(black_box(values)))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_construction, bench_access);
criterion_main!(benches);
