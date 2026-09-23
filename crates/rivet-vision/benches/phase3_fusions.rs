use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use rivet_core::{DType, Device, Tensor};
use rivet_vision::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_vision::transforms::{
    ConvertImageDtypeConfig, CropConfig, FlipConfig, NormalizeConfig, ResizeConfig,
};

fn phase3_fusion_benchmarks(criterion: &mut Criterion) {
    let (batch, height, width, channels) = (16usize, 64usize, 64usize, 3usize);
    let input_values = (0..batch * height * width * channels)
        .map(|index| (index % 256) as u8)
        .collect::<Vec<_>>();
    let input =
        Tensor::from_vec(input_values, [batch, height, width, channels], &Device::Cpu).unwrap();
    let normalize = NormalizeConfig::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]);
    let convert = ConvertImageDtypeConfig::new(DType::F32);
    let layout = rivet_vision::transforms::LayoutConfig::new(ImageAxisOrder::Chw);

    criterion.bench_function("convert_then_normalize", |bencher| {
        bencher.iter(|| {
            let converted = convert
                .apply_batch(black_box(input.clone()), ImageAxisOrder::Hwc)
                .unwrap();
            black_box(
                normalize
                    .apply_batch(converted, ImageAxisOrder::Hwc)
                    .unwrap(),
            )
        });
    });
    criterion.bench_function("late_promotion_normalize_u8", |bencher| {
        bencher.iter(|| {
            black_box(
                normalize
                    .apply_batch(black_box(input.clone()), ImageAxisOrder::Hwc)
                    .unwrap(),
            )
        });
    });
    criterion.bench_function("convert_normalize_layout", |bencher| {
        bencher.iter(|| {
            let converted = convert
                .apply_batch(black_box(input.clone()), ImageAxisOrder::Hwc)
                .unwrap();
            let normalized = normalize
                .apply_batch(converted, ImageAxisOrder::Hwc)
                .unwrap();
            black_box(layout.apply_batch(normalized, ImageAxisOrder::Hwc).unwrap())
        });
    });
    criterion.bench_function("normalize_u8_then_layout_view", |bencher| {
        bencher.iter(|| {
            let normalized = normalize
                .apply_batch(black_box(input.clone()), ImageAxisOrder::Hwc)
                .unwrap();
            black_box(layout.apply_batch(normalized, ImageAxisOrder::Hwc).unwrap())
        });
    });
    criterion.bench_function("fused_normalize_to_chw", |bencher| {
        bencher.iter(|| {
            black_box(
                normalize
                    .apply_batch_to_chw(black_box(input.clone()))
                    .unwrap(),
            )
        });
    });

    // These two references record the current legal execution cost. The
    // planner deliberately does not apply a fused rewrite to either sequence.
    let sample_values = (0..height * width * channels)
        .map(|index| (index % 256) as u8)
        .collect::<Vec<_>>();
    let sample_tensor =
        Tensor::from_vec(sample_values, [height, width, channels], &Device::Cpu).unwrap();
    let crop = CropConfig::new(8, 8, 48, 48);
    let resize = ResizeConfig::new(64, 64);
    criterion.bench_function("crop_then_resize_reference", |bencher| {
        bencher.iter(|| {
            let sample = ImageSample::Decoded(DecodedSample {
                image: sample_tensor.clone(),
                label: 0,
            });
            let cropped = crop.apply(black_box(sample), ImageAxisOrder::Hwc).unwrap();
            black_box(resize.apply(cropped).unwrap())
        });
    });
    let flip = FlipConfig::horizontal();
    criterion.bench_function("flip_then_normalize_reference", |bencher| {
        bencher.iter(|| {
            let sample = ImageSample::Decoded(DecodedSample {
                image: sample_tensor.clone(),
                label: 0,
            });
            let flipped = flip
                .apply(black_box(sample))
                .unwrap()
                .into_decoded()
                .unwrap();
            let batch = Tensor::stack(&[&flipped.image], 0).unwrap();
            black_box(normalize.apply_batch(batch, ImageAxisOrder::Hwc).unwrap())
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = phase3_fusion_benchmarks
}
criterion_main!(benches);
