use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rivet_core::{DType, Device, Tensor};
use std::hint::black_box;

const HEIGHT: usize = 224;
const WIDTH: usize = 224;
const CHANNELS: usize = 3;

fn cuda_batch_normalize(c: &mut Criterion) {
    let device = match Device::cuda(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping CUDA batch normalize benchmark: {error}");
            return;
        }
    };
    let mut group = c.benchmark_group("cuda_u8_nhwc_to_normalized_f32_nchw_224");
    group.sample_size(10);
    for batch in [32usize, 64, 128, 256] {
        let host = Tensor::zeros([batch, HEIGHT, WIDTH, CHANNELS], DType::U8, &Device::Cpu)
            .expect("allocate host U8 input");
        let input = host.to_device(&device).expect("upload U8 input");
        let scale = [
            1.0 / (255.0 * 0.229),
            1.0 / (255.0 * 0.224),
            1.0 / (255.0 * 0.225),
        ];
        let bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225];

        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, _| {
            b.iter(|| {
                let output = input
                    .cuda_normalize_u8_nhwc_to_nchw_f32(&scale, &bias)
                    .expect("launch fused CUDA normalize/layout kernel");
                output.synchronize().expect("wait for fused CUDA kernel");
                black_box(output);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, cuda_batch_normalize);
criterion_main!(benches);
