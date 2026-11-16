use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rivet_core::{DType, Device, Tensor};
use rivet_vision::api::*;
use std::{hint::black_box, time::Duration};

fn sample(input: &Tensor) -> ImageSample {
    ImageSample::Decoded(DecodedSample {
        image: input.clone(),
        label: 0,
    })
}

fn transforms(criterion: &mut Criterion) {
    for size in [32usize, 224, 512] {
        // Benchmark the nonzero offsets used by the decoded slab cache.
        let input = Tensor::from_vec(
            (0..2 * size * size * 3)
                .map(|i| ((i * 17 + i / 31) % 256) as u8)
                .collect::<Vec<_>>(),
            [2, size, size, 3],
            &Device::Cpu,
        )
        .unwrap()
        .get(1)
        .unwrap();
        let mut group = criterion.benchmark_group(format!("transforms/{size}"));
        group.throughput(Throughput::Elements(1));
        macro_rules! run {
            ($name:expr, $body:expr) => {
                group.bench_function(BenchmarkId::from_parameter($name), |b| {
                    b.iter(|| black_box($body))
                });
            };
        }
        run!(
            "brightness",
            BrightnessConfig::new(1.3)
                .apply(black_box(sample(&input)))
                .unwrap()
        );
        run!(
            "contrast",
            ContrastConfig::new(1.3)
                .apply(black_box(sample(&input)))
                .unwrap()
        );
        run!(
            "posterize",
            PosterizeConfig::new(4)
                .apply(black_box(sample(&input)), ImageAxisOrder::Hwc)
                .unwrap()
        );
        run!(
            "autocontrast",
            AutocontrastConfig::new()
                .apply(black_box(sample(&input)), ImageAxisOrder::Hwc)
                .unwrap()
        );
        run!(
            "equalize",
            EqualizeConfig::new()
                .apply(black_box(sample(&input)), ImageAxisOrder::Hwc)
                .unwrap()
        );
        run!(
            "sharpness",
            SharpnessConfig::new(1.5)
                .apply(black_box(sample(&input)), ImageAxisOrder::Hwc)
                .unwrap()
        );
        run!(
            "convert_f32",
            ConvertImageDtypeConfig::new(DType::F32)
                .apply(black_box(sample(&input)), ImageAxisOrder::Hwc)
                .unwrap()
        );
        run!(
            "flip",
            FlipConfig::horizontal()
                .apply(black_box(sample(&input)))
                .unwrap()
        );
        run!(
            "pad4",
            PadConfig::new(4)
                .apply(black_box(sample(&input)), ImageAxisOrder::Hwc)
                .unwrap()
        );
        run!(
            "resize_half",
            ResizeConfig::new(size as u32 / 2, size as u32 / 2)
                .apply(black_box(sample(&input)))
                .unwrap()
        );
        let mut rng = RandomContext::new(42)
            .stream(SampleKey::from_index(0), OpKey::from_parts("RandomCrop", 0));
        run!(
            "random_crop_pad4",
            RandomCropConfig::new(size as u32, size as u32, 4)
                .apply(black_box(sample(&input)), &mut rng, ImageAxisOrder::Hwc)
                .unwrap()
        );
        group.finish();
    }
}
criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));
    targets = transforms
}
criterion_main!(benches);
