use super::{ImageOptimizationOptions, ImagePipeline};
use crate::cache::DenseImageMemoryDataset;
use crate::sample::image::ImageBatch;
use crate::source::ImageSource;
use rivet_core::{DType, Device, Tensor};
use std::sync::Arc;

fn pipeline(workers: usize) -> ImagePipeline {
    let images = Tensor::from_vec(
        (0..16 * 4 * 5 * 3)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>(),
        [16, 4, 5, 3],
        &Device::Cpu,
    )
    .unwrap();
    let labels = Tensor::from_vec((0..16i64).collect::<Vec<_>>(), [16], &Device::Cpu).unwrap();
    let dataset = Arc::new(DenseImageMemoryDataset::new(images, labels).unwrap());
    ImagePipeline::from_source(ImageSource::from_dense_decoded(dataset))
        .workers(workers)
        .seed(71)
}

fn drain(mut loader: crate::runtime::ImageDataLoader) -> Vec<ImageBatch> {
    let mut batches = Vec::new();
    while let Some(batch) = loader.next_batch().unwrap() {
        batches.push(batch);
    }
    batches
}

fn same_f32(left: &[ImageBatch], right: &[ImageBatch]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.images.dims(), right.images.dims());
        assert_eq!(
            left.images.to_vec::<f32>().unwrap(),
            right.images.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            left.labels.to_vec::<i64>().unwrap(),
            right.labels.to_vec::<i64>().unwrap()
        );
    }
}

#[test]
fn canonicalization_options_preserve_executed_slices_layouts_and_labels() {
    for workers in [0, 3] {
        let builder = pipeline(workers)
            .skip(2)
            .take(8)
            .skip(3)
            .take(usize::MAX)
            .hwc_to_chw()
            .chw_to_hwc()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(3, false);
        let optimized = builder.clone().compile().unwrap();
        let disabled = builder
            .clone()
            .compile_with_options(ImageOptimizationOptions {
                common_subplan_elimination: false,
                canonicalize_selections: false,
                canonicalize_layouts: false,
                ..Default::default()
            })
            .unwrap();
        assert!(optimized.info.batch_op_count() < disabled.info.batch_op_count());
        let optimized = drain(optimized);
        same_f32(&optimized, &drain(disabled));
        let labels = optimized
            .iter()
            .flat_map(|batch| batch.labels.to_vec::<i64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(labels, [5, 6, 7, 8, 9]);
        let explain = builder
            .optimization_explain(ImageOptimizationOptions::default())
            .unwrap();
        assert!(explain.contains("Declared logical graph:"));
        assert!(explain.contains("Optimized logical graph:"));
    }
}

#[test]
fn default_float_precision_matches_unoptimized_conversion_bit_for_bit() {
    for workers in [0, 3] {
        let original = pipeline(workers)
            .convert_image_dtype(DType::F32)
            .normalize(vec![0.13; 3], vec![0.37; 3])
            .batch(4, false);
        let strict = drain(original.clone().compile().unwrap());
        let reference = drain(original.clone().compile_unoptimized_for_test(0).unwrap());
        same_f32(&strict, &reference);
        let strict_explain = original
            .optimization_explain(ImageOptimizationOptions::default())
            .unwrap();
        assert!(!strict_explain.contains("rewrite.dtype-late-promotion"));
        let options = ImageOptimizationOptions {
            allow_float_reassociation: true,
            ..Default::default()
        };
        let reassociated = drain(original.clone().compile_with_options(options).unwrap());
        let mut changed_bits = false;
        for (left, right) in strict.iter().zip(&reassociated) {
            assert_eq!(
                left.labels.to_vec::<i64>().unwrap(),
                right.labels.to_vec::<i64>().unwrap()
            );
            for (left, right) in left
                .images
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .zip(right.images.to_vec::<f32>().unwrap())
            {
                assert!((left - right).abs() <= 1e-6);
                changed_bits |= left.to_bits() != right.to_bits();
            }
        }
        assert!(
            changed_bits,
            "fixture must exercise float rounding differences"
        );
        assert!(
            original
                .optimization_explain(options)
                .unwrap()
                .contains("rewrite.dtype-late-promotion")
        );
    }
}

#[test]
fn snapshots_and_incremental_properties_do_not_change_optimized_graph() {
    for workers in [0, 3] {
        let builder = pipeline(workers)
            .horizontal_flip()
            .crop(1, 1, 3, 2)
            .take(8)
            .shuffle(91)
            .skip(2)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(3, false);
        let mut normal = builder.to_logical_plan();
        let mut traced = normal.clone();
        let (normal_context, normal_placement) =
            super::optimizer::optimize_vision_plan_with_options(
                &mut normal,
                workers,
                ImageOptimizationOptions::default(),
            )
            .unwrap();
        let (traced_context, traced_placement) =
            super::optimizer::optimize_vision_plan_with_options(
                &mut traced,
                workers,
                ImageOptimizationOptions {
                    record_snapshots: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(normal.explain().unwrap(), traced.explain().unwrap());
        assert_eq!(normal_placement, traced_placement);
        assert!(normal_context.snapshots.is_empty());
        assert!(!traced_context.snapshots.is_empty());
        assert_eq!(
            normal_context
                .diagnostics
                .iter()
                .filter(|d| d.code == "placement.explain")
                .count(),
            1
        );
        let full = normal
            .infer_properties(&super::inference::VisionPropertyInference::new(workers))
            .unwrap();
        for (id, properties) in normal_context.annotations().iter() {
            assert_eq!(Some(properties), full.get(id));
        }
        let (direct, _) = super::optimizer::optimize_vision_plan_with_options(
            &mut traced,
            workers,
            ImageOptimizationOptions::default(),
        )
        .unwrap();
        for (id, properties) in normal_context.annotations().iter() {
            assert_eq!(Some(properties), direct.annotations().get(id));
        }
    }
}

#[test]
fn identity_dtype_is_not_needed_as_a_worker_scheduling_barrier() {
    for workers in [0, 3] {
        let builder = pipeline(workers)
            .convert_image_dtype(DType::U8)
            .crop(1, 1, 3, 2)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false);
        let optimized = drain(builder.clone().compile().unwrap());
        let reference = drain(builder.clone().compile_unoptimized_for_test(0).unwrap());
        same_f32(&optimized, &reference);
        assert_eq!(optimized[0].images.dims(), [4, 3, 2, 3]);
        assert!(
            builder
                .optimization_explain(ImageOptimizationOptions::default())
                .unwrap()
                .contains("rewrite.identity-dtype")
        );
    }
}
