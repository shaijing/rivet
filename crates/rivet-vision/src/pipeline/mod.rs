mod builder;
mod compile;
pub mod inference;
mod logical;
pub mod op;
mod optimizer;
mod transform;

pub use builder::ImagePipeline;
pub use transform::{Compose, ImageTransform, TransformSequence};

#[cfg(test)]
mod tests {
    use super::{ImagePipeline, TransformSequence};
    use crate::cache::DenseImageMemoryDataset;
    use crate::pipeline::op::{ExecutionKind, PipelineImageState, SampleKernel};
    use crate::sample::image::{DecodedSample, EncodedImageSample, ImageAxisOrder};
    use crate::source::ImageSource;
    use crate::transforms::Point2;
    #[cfg(feature = "cuda")]
    use crate::transforms::{InterpolationMode, RandomResizedCropConfig};
    use arrow_buffer::Buffer;
    use rivet_core::{DType, Device, Tensor};
    use rivet_data::dataset::Dataset;
    use std::sync::Arc;

    struct StubDataset {
        len: usize,
    }

    impl Dataset for StubDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.len
        }

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            Ok(indices
                .iter()
                .map(|_| EncodedImageSample {
                    image: Buffer::from(Vec::<u8>::new()),
                    label: 0,
                })
                .collect())
        }
    }

    struct DecodedStubDataset;

    impl Dataset for DecodedStubDataset {
        type Item = DecodedSample;

        fn len(&self) -> usize {
            1
        }

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            indices
                .iter()
                .map(|&index| {
                    if index != 0 {
                        return Err(rivet_data::DataError::IndexOutOfRange { index, len: 1 });
                    }
                    Ok(DecodedSample {
                        image: Tensor::from_vec(vec![255u8, 0, 0], [1, 1, 3], &Device::Cpu)
                            .unwrap(),
                        label: 7,
                    })
                })
                .collect()
        }
    }

    fn stub(len: usize) -> ImagePipeline {
        ImagePipeline::new(Arc::new(StubDataset { len }))
    }

    fn decoded_stub() -> ImagePipeline {
        ImagePipeline::from_source(ImageSource::from_decoded(Arc::new(DecodedStubDataset)))
    }

    fn dense_source(batch_native: bool) -> ImageSource {
        let dataset = Arc::new(
            DenseImageMemoryDataset::new(
                Tensor::from_vec(
                    vec![10u8, 11, 12, 20, 21, 22, 30, 31, 32],
                    [3, 1, 1, 3],
                    &Device::Cpu,
                )
                .unwrap(),
                Tensor::from_vec(vec![10i64, 20, 30], [3], &Device::Cpu).unwrap(),
            )
            .unwrap(),
        );
        if batch_native {
            ImageSource::from_dense_decoded(dataset)
        } else {
            ImageSource::from_decoded(dataset)
        }
    }

    #[cfg(feature = "cuda")]
    fn spatial_dense_source() -> ImageSource {
        let mut images = Vec::with_capacity(5 * 4 * 4 * 3);
        for index in 0..5u8 {
            for y in 0..4u8 {
                for x in 0..4u8 {
                    images.extend_from_slice(&[
                        index * 17 + y * 13 + x * 3,
                        index * 11 + y * 7 + x * 5,
                        index * 5 + y * 3 + x * 9,
                    ]);
                }
            }
        }
        let dataset = Arc::new(
            DenseImageMemoryDataset::new(
                Tensor::from_vec(images, [5, 4, 4, 3], &Device::Cpu).unwrap(),
                Tensor::from_vec((0i64..5).collect(), [5], &Device::Cpu).unwrap(),
            )
            .unwrap(),
        );
        ImageSource::from_dense_decoded(dataset)
    }

    fn compile_err(pipeline: ImagePipeline) -> String {
        match pipeline.compile() {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected a compile error"),
        }
    }

    fn legacy_compile_err(pipeline: ImagePipeline) -> String {
        match pipeline.compile_legacy_for_test(0) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected a legacy compile error"),
        }
    }

    #[test]
    fn resize_before_decode_rejected_at_compile() {
        let err = compile_err(stub(10).resize(8, 8).batch(4, false));
        assert!(
            err.contains("Resize requires a decoded image"),
            "got: {err}"
        );
    }

    #[test]
    fn normalize_before_decode_rejected_at_compile() {
        let err = compile_err(
            stub(10)
                .normalize(vec![0.5; 3], vec![0.5; 3])
                .batch(4, false),
        );
        assert!(
            err.contains("Normalize requires a decoded image"),
            "got: {err}"
        );
    }

    #[test]
    fn double_decode_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().decode_image().batch(4, false));
        assert!(
            err.contains("Decode requires an encoded image"),
            "got: {err}"
        );
    }

    #[test]
    fn batching_while_encoded_rejected_at_compile() {
        let err = compile_err(stub(10).batch(4, false));
        assert!(
            err.contains("must decode images before batching"),
            "got: {err}"
        );
    }

    #[test]
    fn decoded_source_can_batch_without_decode_op() {
        let mut loader = decoded_stub().batch(1, false).compile().unwrap();
        let batch = loader.next_batch().unwrap().unwrap();

        assert_eq!(batch.labels.to_vec::<i64>().unwrap(), [7]);
        assert_eq!(batch.images.dims(), [1, 1, 1, 3]);
    }

    #[test]
    fn decoded_source_rejects_decode_op() {
        let err = compile_err(decoded_stub().decode_image().batch(1, false));

        assert!(
            err.contains("Decode requires an encoded image"),
            "got: {err}"
        );
    }

    #[test]
    fn invalid_resize_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().resize(0, 8).batch(4, false));
        assert!(
            err.contains("resize width and height must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn random_crop_before_decode_rejected_at_compile() {
        let err = compile_err(stub(10).random_crop(8, 8, 4).batch(4, false));
        assert!(
            err.contains("RandomCrop requires a decoded image"),
            "got: {err}"
        );
    }

    #[test]
    fn invalid_random_flip_probability_rejected_at_compile() {
        let err = compile_err(
            stub(10)
                .decode_image()
                .random_horizontal_flip(1.5)
                .batch(4, false),
        );
        assert!(
            err.contains("probability must be in [0.0, 1.0]"),
            "got: {err}"
        );
    }

    #[test]
    fn invalid_random_crop_size_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().random_crop(0, 8, 4).batch(4, false));
        assert!(
            err.contains("width and height must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn invalid_normalize_rejected_at_compile() {
        let err = compile_err(
            stub(10)
                .decode_image()
                .normalize(vec![0.5; 3], vec![0.5; 2])
                .batch(4, false),
        );
        assert!(err.contains("same length"), "got: {err}");
    }

    #[test]
    fn phase1_representation_and_color_ops_update_pipeline_state() {
        let loader = stub(1)
            .decode_image()
            .grayscale(1)
            .convert_image_dtype(DType::F32)
            .batch(1, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 2);
        assert_eq!(loader.plan.batch_ops.len(), 1);
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::F32,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
    }

    #[test]
    fn phase1_invalid_configs_are_rejected_at_compile() {
        let err = compile_err(stub(1).decode_image().grayscale(2).batch(1, false));
        assert!(err.contains("must be 1 or 3"), "got: {err}");

        let err = compile_err(stub(1).decode_image().contrast(f32::NAN).batch(1, false));
        assert!(err.contains("must be finite"), "got: {err}");
    }

    #[test]
    fn phase2_sample_ops_compile_before_batching() {
        let loader = stub(1)
            .decode_image()
            .pad(1)
            .random_resized_crop(8, 8)
            .color_jitter(4, 0.2, 10)
            .gaussian_blur(0.5)
            .random_grayscale(0.0, 3)
            .random_erasing(0.0)
            .batch(1, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 7);
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
    }

    #[test]
    fn phase2_invalid_configs_are_rejected_at_compile() {
        let err = compile_err(
            stub(1)
                .decode_image()
                .random_resized_crop(0, 8)
                .batch(1, false),
        );
        assert!(
            err.contains("width and height must be greater than 0"),
            "got: {err}"
        );

        let err = compile_err(stub(1).decode_image().gaussian_blur(-1.0).batch(1, false));
        assert!(err.contains("finite and non-negative"), "got: {err}");

        let err = compile_err(
            stub(1)
                .decode_image()
                .color_jitter(-1, 0.0, 0)
                .batch(1, false),
        );
        assert!(
            err.contains("brightness/hue must be non-negative"),
            "got: {err}"
        );
    }

    #[test]
    fn phase3_u8_color_ops_compile_before_batching() {
        let loader = stub(1)
            .decode_image()
            .invert()
            .posterize(4)
            .solarize(100)
            .autocontrast()
            .equalize()
            .sharpness(0.5)
            .batch(1, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 7);
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
    }

    #[test]
    fn phase3_u8_color_ops_run_on_decoded_samples() {
        let mut loader = decoded_stub()
            .invert()
            .posterize(4)
            .solarize(100)
            .autocontrast()
            .equalize()
            .sharpness(0.0)
            .batch(1, false)
            .compile()
            .unwrap();
        let batch = loader.next_batch().unwrap().unwrap();

        assert_eq!(batch.labels.to_vec::<i64>().unwrap(), [7]);
        assert_eq!(batch.images.dims(), [1, 1, 1, 3]);
        assert_eq!(batch.images.to_vec::<u8>().unwrap(), [0, 15, 15]);
    }

    #[test]
    fn phase3_invalid_configs_are_rejected_at_compile() {
        let err = compile_err(stub(1).decode_image().posterize(0).batch(1, false));
        assert!(
            err.contains("posterize bits must be in [1, 8]"),
            "got: {err}"
        );

        let err = compile_err(stub(1).decode_image().sharpness(-1.0).batch(1, false));
        assert!(
            err.contains("sharpness amount must be finite and non-negative"),
            "got: {err}"
        );
    }

    #[test]
    fn phase4_composition_controls_compile_into_sample_stage() {
        let loader = stub(1)
            .decode_image()
            .compose(TransformSequence::new().brightness(2).invert())
            .random_apply(0.5, TransformSequence::new().horizontal_flip())
            .random_choice(vec![
                TransformSequence::new().contrast(1.0),
                TransformSequence::new().solarize(100),
            ])
            .random_order(TransformSequence::new().brightness(1).invert())
            .batch(1, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 6);
        assert_eq!(loader.plan.batch_ops.len(), 0);
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
    }

    #[test]
    fn phase3_nested_controls_lower_to_compiled_programs() {
        let loader = stub(1)
            .decode_image()
            .random_apply(
                0.5,
                TransformSequence::new().random_choice(vec![
                    TransformSequence::new().brightness(2),
                    TransformSequence::new()
                        .random_order(TransformSequence::new().invert().contrast(1.0)),
                ]),
            )
            .batch(1, false)
            .compile()
            .unwrap();

        let SampleKernel::RandomApply { body, .. } = &loader.plan.sample_ops[1].kernel else {
            panic!("expected RandomApply compiled kernel");
        };
        let SampleKernel::RandomChoice { branches, .. } = &body.ops[0].kernel else {
            panic!("expected RandomChoice compiled kernel");
        };
        assert!(matches!(
            &branches[0].ops[0].kernel,
            SampleKernel::Semantic(_)
        ));

        let SampleKernel::RandomOrder { ops, .. } = &branches[1].ops[0].kernel else {
            panic!("expected RandomOrder compiled kernel");
        };
        assert_eq!(ops.len(), 2);
        assert!(
            ops.iter()
                .all(|op| matches!(&op.kernel, SampleKernel::Semantic(_)))
        );
    }

    #[test]
    fn phase4_composition_controls_validate_nested_contracts() {
        let err = compile_err(
            stub(1)
                .decode_image()
                .random_apply(1.5, TransformSequence::new().invert())
                .batch(1, false),
        );
        assert!(err.contains("random_apply probability"), "got: {err}");

        let err = compile_err(
            stub(1)
                .decode_image()
                .random_choice(Vec::new())
                .batch(1, false),
        );
        assert!(err.contains("at least one choice"), "got: {err}");

        let err = compile_err(
            stub(1)
                .decode_image()
                .random_order(TransformSequence::new().normalize(vec![0.0; 3], vec![1.0; 3]))
                .batch(1, false),
        );
        assert!(err.contains("sample-stage operations"), "got: {err}");

        let err = compile_err(
            stub(1)
                .decode_image()
                .random_apply(
                    0.5,
                    TransformSequence::new().convert_image_dtype(DType::F32),
                )
                .batch(1, false),
        );
        assert!(err.contains("is batch-stage"), "got: {err}");
    }

    #[test]
    fn phase5_advanced_geometry_compiles_as_sample_ops() {
        let points = [
            Point2::new(0.0, 0.0),
            Point2::new(7.0, 0.0),
            Point2::new(7.0, 7.0),
            Point2::new(0.0, 7.0),
        ];
        let loader = stub(1)
            .decode_image()
            .arbitrary_rotate(15.0)
            .random_affine(10.0)
            .perspective(points, points)
            .random_perspective(0.2, 0.5)
            .elastic_transform(1.0, 1.0)
            .batch(1, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 6);
        assert_eq!(loader.plan.batch_ops.len(), 0);
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
    }

    #[test]
    fn phase5_advanced_geometry_rejects_invalid_configs() {
        let err = compile_err(
            stub(1)
                .decode_image()
                .arbitrary_rotate(f32::NAN)
                .batch(1, false),
        );
        assert!(err.contains("angle must be finite"), "got: {err}");

        let err = compile_err(stub(1).decode_image().random_affine(-1.0).batch(1, false));
        assert!(err.contains("degrees must be finite"), "got: {err}");

        let err = compile_err(
            stub(1)
                .decode_image()
                .random_perspective(1.1, 0.5)
                .batch(1, false),
        );
        assert!(err.contains("distortion_scale"), "got: {err}");

        let err = compile_err(
            stub(1)
                .decode_image()
                .elastic_transform(1.0, -1.0)
                .batch(1, false),
        );
        assert!(err.contains("sigma must be finite"), "got: {err}");
    }

    #[test]
    fn deterministic_ops_do_not_perturb_random_op_keys() {
        let base = stub(1)
            .decode_image()
            .random_crop(1, 1, 0)
            .random_horizontal_flip(0.5)
            .batch(1, false)
            .compile()
            .unwrap();
        let with_deterministic = stub(1)
            .decode_image()
            .resize(1, 1)
            .random_crop(1, 1, 0)
            .brightness(0)
            .random_horizontal_flip(0.5)
            .batch(1, false)
            .compile()
            .unwrap();

        assert_eq!(base.plan.sample_ops[0].random_key, None);
        assert_eq!(
            base.plan.sample_ops[1].random_key,
            with_deterministic.plan.sample_ops[2].random_key
        );
        assert_eq!(
            base.plan.sample_ops[2].random_key,
            with_deterministic.plan.sample_ops[4].random_key
        );
        assert_ne!(
            base.plan.sample_ops[1].random_key,
            base.plan.sample_ops[2].random_key
        );
    }

    #[test]
    fn zero_batch_size_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().batch(0, false));
        assert!(
            err.contains("batch size must be greater than 0"),
            "got: {err}"
        );
    }

    #[test]
    fn compile_reports_output_state() {
        let loader = stub(10)
            .decode_image()
            .resize(8, 8)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false)
            .compile()
            .unwrap();

        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::F32,
                axis_order: ImageAxisOrder::Chw,
            }
        );
    }

    #[test]
    fn compile_splits_sample_and_batch_stages() {
        let loader = stub(10)
            .decode_image()
            .resize(8, 8)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 2);
        assert_eq!(loader.plan.batch_ops.len(), 1);
        assert_eq!(loader.plan.input_state, PipelineImageState::Encoded);
        assert_eq!(
            loader.plan.pre_batch_state,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::F32,
                axis_order: ImageAxisOrder::Chw,
            }
        );
        assert_eq!(
            loader.plan.sample_ops[0].execution_kind(),
            ExecutionKind::Sample
        );
        assert_eq!(
            loader.plan.batch_ops[0].execution_kind(),
            ExecutionKind::Batch
        );
        assert_eq!(loader.plan.batch_ops[0].name(), "NormalizeToChw");
    }

    #[test]
    fn workers_keep_normalization_on_sample_stage() {
        let loader = stub(10)
            .decode_image()
            .resize(8, 8)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .workers(2)
            .batch(4, false)
            .compile()
            .unwrap();

        assert_eq!(loader.plan.sample_ops.len(), 3);
        assert_eq!(loader.plan.sample_ops[2].name(), "NormalizeSample");
        assert_eq!(loader.plan.batch_ops.len(), 1);
        assert_eq!(loader.plan.batch_ops[0].name(), "Layout");
        assert_eq!(
            loader.plan.pre_batch_state,
            PipelineImageState::Decoded {
                dtype: DType::F32,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
    }

    #[test]
    fn dtype_conversion_runs_after_sample_stack() {
        let mut loader = decoded_stub()
            .convert_image_dtype(DType::F32)
            .batch(1, false)
            .compile()
            .unwrap();

        assert!(loader.plan.sample_ops.is_empty());
        assert_eq!(loader.plan.batch_ops.len(), 1);
        assert_eq!(
            loader.plan.pre_batch_state,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            }
        );
        assert_eq!(
            loader.plan.output_state,
            PipelineImageState::Decoded {
                dtype: DType::F32,
                axis_order: ImageAxisOrder::Hwc,
            }
        );

        let batch = loader.next_batch().unwrap().unwrap();
        assert_eq!(batch.images.to_vec::<f32>().unwrap(), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn compiler_removes_noop_layout_transition() {
        let loader = decoded_stub()
            .chw_to_hwc()
            .batch(1, false)
            .compile()
            .unwrap();

        assert!(loader.plan.sample_ops.is_empty());
        assert!(loader.plan.batch_ops.is_empty());
        assert_eq!(loader.plan.pre_batch_state, loader.plan.input_state);
        assert_eq!(loader.plan.output_state, loader.plan.input_state);
    }

    #[test]
    fn sample_ops_cannot_follow_batch_stage() {
        let err = compile_err(
            stub(10)
                .decode_image()
                .normalize(vec![0.5; 3], vec![0.5; 3])
                .resize(8, 8)
                .batch(4, false),
        );
        assert!(
            err.contains("Resize cannot follow the batch stage"),
            "got: {err}"
        );

        let err = compile_err(
            stub(10)
                .decode_image()
                .convert_image_dtype(DType::F32)
                .brightness(1)
                .batch(4, false),
        );
        assert!(
            err.contains("Brightness cannot follow the batch stage"),
            "got: {err}"
        );
    }

    #[test]
    fn batch_stage_runs_after_sample_stack() {
        let mut loader = decoded_stub()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(1, false)
            .compile()
            .unwrap();
        let batch = loader.next_batch().unwrap().unwrap();

        assert_eq!(batch.images.dims(), [1, 3, 1, 1]);
        assert_eq!(batch.images.dtype(), DType::F32);
        let values = batch.images.to_vec::<f32>().unwrap();
        assert_eq!(values, [1.0, -1.0, -1.0]);
    }

    #[test]
    fn dense_batch_native_read_matches_decoded_fallback() {
        let mut direct = ImagePipeline::from_source(dense_source(true))
            .normalize(vec![0.0; 3], vec![1.0; 3])
            .hwc_to_chw()
            .workers(4)
            .batch(2, false)
            .compile()
            .unwrap();
        let mut fallback = ImagePipeline::from_source(dense_source(false))
            .normalize(vec![0.0; 3], vec![1.0; 3])
            .hwc_to_chw()
            .workers(4)
            .batch(2, false)
            .compile()
            .unwrap();

        assert!(direct.plan.can_use_batch_native());
        assert!(!fallback.plan.can_use_batch_native());

        for _ in 0..2 {
            let direct_batch = direct.next_batch().unwrap().unwrap();
            let fallback_batch = fallback.next_batch().unwrap().unwrap();
            assert_eq!(
                direct_batch.labels.to_vec::<i64>().unwrap(),
                fallback_batch.labels.to_vec::<i64>().unwrap()
            );
            assert_eq!(
                direct_batch.images.dims(),
                [direct_batch.labels.dims()[0], 3, 1, 1]
            );
            assert_eq!(direct_batch.images.dims(), fallback_batch.images.dims());
            assert_eq!(
                direct_batch.images.to_vec::<f32>().unwrap(),
                fallback_batch.images.to_vec::<f32>().unwrap()
            );
        }

        assert!(direct.next_batch().unwrap().is_none());
        assert!(fallback.next_batch().unwrap().is_none());
    }

    #[test]
    fn image_pipeline_logical_round_trip_preserves_builder_configuration() {
        let original = stub(10)
            .skip(2)
            .take(5)
            .shuffle(17)
            .decode_image()
            .resize(8, 6)
            .random_horizontal_flip(0.25)
            .normalize(vec![0.1, 0.2, 0.3], vec![1.0; 3])
            .hwc_to_chw()
            .workers(3)
            .prefetch_batches(4)
            .stage_queue_max_bytes(16_384)
            .seed(91)
            .epoch(7)
            .batch(2, true);

        let logical = original.to_logical_plan();
        let explain = logical.explain().unwrap();
        assert!(explain.contains("LogicalPlan(root="));
        assert!(explain.contains("Source vision::Source"));
        assert!(explain.contains("Op vision::ImageOp"));
        assert!(explain.contains("Batch vision::BatchConfig"));
        assert!(explain.contains("Sink <-"));

        let restored = ImagePipeline::from_logical_plan(&logical).unwrap();
        assert_eq!(restored.index_ops.len(), 3);
        assert_eq!(
            restored.ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
            [
                "Decode",
                "Resize",
                "RandomHorizontalFlip",
                "Normalize",
                "Layout"
            ]
        );
        assert_eq!(restored.batch.unwrap().size, 2);
        assert!(restored.batch.unwrap().drop_last);
        assert_eq!(restored.runtime.num_workers, 3);
        assert_eq!(restored.runtime.prefetch_batches, 4);
        assert_eq!(restored.runtime.stage_queue_max_bytes, 16_384);
        assert_eq!(restored.epoch, 7);
        assert_eq!(restored.global_seed, Some(91));
    }

    #[test]
    fn optimizer_discovers_legal_normalize_layout_fusion() {
        let pipeline = stub(4)
            .decode_image()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(2, false);
        let mut plan = pipeline.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fusion.legal" && diagnostic.message.contains("normalize-layout")
        }));
        assert!(
            context
                .snapshots
                .iter()
                .any(|(name, _)| name == "fusion-discovery")
        );
        assert!(plan.nodes().any(|(_, node)| {
            node.payload()
                .is_some_and(|payload| payload.name() == "FusionGroup")
        }));
        assert!(
            context.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "fusion.applied" && diagnostic.message.contains("NormalizeToChw")
            }),
            "diagnostics: {:?}",
            context.diagnostics
        );
        let explain = pipeline
            .placement_explain(rivet_plan::MachineProfile::default())
            .unwrap();
        assert!(explain.contains("physical candidate"));
        assert!(explain.contains("vision-normalize-to-chw-fused"));
    }

    #[test]
    fn late_dtype_promotion_matches_explicit_convert_then_normalize() {
        let original = decoded_stub()
            .convert_image_dtype(DType::F32)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(1, false);
        let mut plan = original.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code == "rewrite.dtype-late-promotion" })
        );
        assert!(!plan.nodes().any(|(_, node)| {
            node.payload()
                .is_some_and(|payload| payload.name() == "ImageOp")
                && node
                    .payload_as::<crate::pipeline::op::ImageOp>()
                    .is_some_and(|op| op.name() == "ConvertImageDtype")
        }));

        let optimized = ImagePipeline::from_logical_plan(&plan).unwrap();
        assert_eq!(
            optimized.ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
            ["Normalize"]
        );
        let mut baseline_loader = original.compile_legacy_for_test(0).unwrap();
        let mut optimized_loader = optimized.compile().unwrap();
        let baseline = baseline_loader.next_batch().unwrap().unwrap();
        let rewritten = optimized_loader.next_batch().unwrap().unwrap();
        assert_eq!(baseline.images.dims(), rewritten.images.dims());
        assert_eq!(
            baseline.images.to_vec::<f32>().unwrap(),
            rewritten.images.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn late_dtype_promotion_preserves_worker_batch_barrier() {
        let pipeline = stub(1)
            .decode_image()
            .horizontal_flip()
            .convert_image_dtype(DType::F32)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .workers(2)
            .batch(1, false);
        let mut plan = pipeline.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 2).unwrap();
        assert!(
            !context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "rewrite.dtype-late-promotion")
        );
        let lowered = ImagePipeline::from_logical_plan(&plan).unwrap();
        assert_eq!(
            lowered.ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
            ["Decode", "Flip", "ConvertImageDtype", "Normalize"]
        );
    }

    #[test]
    fn identity_dtype_conversion_is_removed() {
        let original = decoded_stub()
            .convert_image_dtype(DType::U8)
            .batch(1, false);
        let mut plan = original.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "rewrite.identity-dtype")
        );
        let optimized = ImagePipeline::from_logical_plan(&plan).unwrap();
        assert!(optimized.ops.is_empty());

        let mut baseline_loader = original.compile_legacy_for_test(0).unwrap();
        let mut optimized_loader = optimized.compile().unwrap();
        let baseline = baseline_loader.next_batch().unwrap().unwrap();
        let rewritten = optimized_loader.next_batch().unwrap().unwrap();
        assert_eq!(baseline.images.dims(), rewritten.images.dims());
        assert_eq!(
            baseline.images.to_vec::<u8>().unwrap(),
            rewritten.images.to_vec::<u8>().unwrap()
        );
    }

    #[test]
    fn convert_normalize_layout_rewrites_to_one_fusion_group_and_matches_legacy() {
        let original = decoded_stub()
            .convert_image_dtype(DType::F32)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(1, false);
        let mut plan = original.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code == "rewrite.dtype-late-promotion" })
        );
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fusion.applied" && diagnostic.message.contains("NormalizeToChw")
        }));
        let fused_id = plan
            .nodes()
            .find_map(|(id, node)| {
                node.payload()
                    .is_some_and(|payload| payload.name() == "FusionGroup")
                    .then_some(id)
            })
            .unwrap();
        let fused_properties = context.annotations().get(fused_id).unwrap();
        assert_eq!(fused_properties.dtype, Some(rivet_plan::DataType::F32));
        assert_eq!(
            fused_properties.axis_order,
            Some(rivet_plan::AxisOrder::Chw)
        );
        assert_eq!(
            fused_properties.contiguity,
            Some(rivet_plan::Contiguity::Contiguous)
        );

        let optimized = ImagePipeline::from_logical_plan(&plan).unwrap();
        let mut baseline_loader = original.compile_legacy_for_test(0).unwrap();
        let mut optimized_loader = optimized.compile().unwrap();
        let baseline = baseline_loader.next_batch().unwrap().unwrap();
        let rewritten = optimized_loader.next_batch().unwrap().unwrap();
        assert_eq!(baseline.images.dims(), rewritten.images.dims());
        assert_eq!(
            baseline.images.to_vec::<f32>().unwrap(),
            rewritten.images.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn fusion_discovery_records_unsafe_or_unimplemented_reorders() {
        let crop_resize = stub(1)
            .decode_image()
            .crop(0, 0, 1, 1)
            .resize(2, 2)
            .batch(1, false);
        let mut plan = crop_resize.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fusion.illegal"
                && diagnostic.message.contains("CropResize")
                && diagnostic
                    .message
                    .contains("interpolation/border semantics")
        }));

        let flip_normalize = decoded_stub()
            .horizontal_flip()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(1, false);
        let mut plan = flip_normalize.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fusion.illegal"
                && diagnostic.message.contains("FlipNormalize")
                && diagnostic.message.contains("no cross-stage fused kernel")
        }));
    }

    #[test]
    fn known_full_image_crop_is_removed_before_following_resize() {
        let original = decoded_stub()
            .resize(1, 1)
            .crop(0, 0, 1, 1)
            .resize(2, 2)
            .batch(1, false);
        let mut plan = original.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "rewrite.identity-crop"
                && diagnostic.message.contains("known input shape")
        }));
        let optimized = ImagePipeline::from_logical_plan(&plan).unwrap();
        assert_eq!(
            optimized.ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
            ["Resize", "Resize"]
        );
        let mut baseline_loader = original.compile_legacy_for_test(0).unwrap();
        let mut optimized_loader = optimized.compile().unwrap();
        let baseline = baseline_loader.next_batch().unwrap().unwrap();
        let rewritten = optimized_loader.next_batch().unwrap().unwrap();
        assert_eq!(baseline.images.dims(), rewritten.images.dims());
        assert_eq!(
            baseline.images.to_vec::<u8>().unwrap(),
            rewritten.images.to_vec::<u8>().unwrap()
        );
    }

    #[test]
    fn skip_take_shuffle_are_compiled_before_source_reads() {
        let pipeline = decoded_stub()
            .skip(0)
            .take(1)
            .shuffle(7)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(1, false);
        let mut plan = pipeline.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "rewrite.index-source-pushdown"
                && diagnostic
                    .message
                    .contains("before reads and image transforms")
        }));
    }

    #[test]
    fn optimizer_rejects_normalize_layout_fusion_after_worker_sample_ops() {
        let pipeline = stub(4)
            .decode_image()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .workers(2)
            .batch(2, false);
        let mut plan = pipeline.to_logical_plan();
        let context = super::optimizer::optimize_vision_plan(&mut plan, 2).unwrap();
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fusion.illegal" && diagnostic.message.contains("normalize-layout")
        }));
    }

    #[test]
    fn placement_profile_changes_parallel_strategy_and_explains_cost() {
        let pipeline = decoded_stub().resize(8, 8).batch(2, false);
        let single = pipeline
            .placement_explain(rivet_plan::MachineProfile {
                cpu_threads: 1,
                ..rivet_plan::MachineProfile::default()
            })
            .unwrap();
        let multi = pipeline
            .placement_explain(rivet_plan::MachineProfile {
                cpu_threads: 4,
                ..rivet_plan::MachineProfile::default()
            })
            .unwrap();
        assert!(single.contains("kernel=vision::ImageOp-cpu-Sample"));
        assert!(single.contains("cost="));
        assert!(single.contains("parallelism=1"));
        assert!(single.contains("physical candidate"));
        assert!(single.contains("vision-image-cpu"));
        assert!(multi.contains("parallelism=4"));
        assert!(multi.contains("source access=RandomAccess"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn placement_rejects_cuda_sink_without_registered_cuda_kernels() {
        let pipeline = decoded_stub()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(2, false);
        let error = pipeline
            .placement_explain(rivet_plan::MachineProfile {
                cpu_threads: 4,
                available_devices: vec![
                    rivet_plan::DeviceClass::Cpu,
                    rivet_plan::DeviceClass::Cuda,
                ],
                preferred_sink_device: Some(rivet_plan::DeviceClass::Cuda),
                ..rivet_plan::MachineProfile::default()
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("no compatible registered kernel"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sink_placement_costs_one_explicit_transfer_boundary() {
        let pipeline = decoded_stub().batch(1, false);
        let explanation = pipeline
            .placement_explain(rivet_plan::MachineProfile {
                cpu_threads: 1,
                available_devices: vec![
                    rivet_plan::DeviceClass::Cpu,
                    rivet_plan::DeviceClass::Cuda,
                ],
                preferred_sink_device: Some(rivet_plan::DeviceClass::Cuda),
                ..rivet_plan::MachineProfile::default()
            })
            .unwrap();
        assert!(explanation.contains("device=Cuda"), "{explanation}");
        assert_eq!(explanation.matches("transfer before %").count(), 1);
        let cuda_sink = explanation
            .lines()
            .find(|line| line.contains("device=Cuda"))
            .expect("CUDA sink placement is present");
        assert!(
            cuda_sink.contains("transfer=") && !cuda_sink.contains("transfer=0B"),
            "{cuda_sink}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sink_returns_shape_dtype_and_values_on_device_after_ordered_h2d() {
        let Ok(_device) = Device::cuda(0) else {
            eprintln!("skipping CUDA sink test because device 0 is unavailable");
            return;
        };
        let mut loader = decoded_stub()
            .batch(1, false)
            .cuda_sink(0)
            .compile()
            .unwrap();
        let explanation = loader.physical_explain().unwrap();
        assert_eq!(explanation.matches("Transfer(H2D)").count(), 1);
        assert!(explanation.contains("target=Device { ordinal: 0 }"));
        assert!(explanation.contains("estimated_transfer_bytes="));

        let batch = loader.next_batch().unwrap().unwrap();
        assert_eq!(batch.images.dims(), [1, 1, 1, 3]);
        assert_eq!(batch.images.dtype(), DType::U8);
        assert_eq!(batch.labels.dims(), [1]);
        assert_eq!(batch.labels.dtype(), DType::I64);
        assert!(batch.images.device().is_cuda());
        assert!(batch.labels.device().is_cuda());
        if let Device::Cuda(device) = batch.images.device() {
            let stats = device.debug_stats();
            assert_eq!(stats.h2d_count, 2);
            assert!(stats.synchronize_count >= 1);
        }
        assert_eq!(batch.images.to_vec::<u8>().unwrap(), [255, 0, 0]);
        assert_eq!(batch.labels.to_vec::<i64>().unwrap(), [7]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sink_runs_fused_nhwc_normalize_to_nchw_batch_kernel() {
        let Ok(_device) = Device::cuda(0) else {
            eprintln!("skipping CUDA fused vision test because device 0 is unavailable");
            return;
        };
        let mean = vec![0.5, 0.25, 0.0];
        let std = vec![0.5, 0.5, 1.0];
        let cpu_pipeline = decoded_stub()
            .normalize(mean.clone(), std.clone())
            .hwc_to_chw()
            .batch(1, false);
        let mut cpu_loader = cpu_pipeline.clone().compile().unwrap();
        let cpu_batch = cpu_loader.next_batch().unwrap().unwrap();

        let mut cuda_loader = cpu_pipeline.cuda_sink(0).compile().unwrap();
        let physical = cuda_loader.physical_explain().unwrap();
        let transfer_at = physical.find("Transfer(H2D)").unwrap();
        let kernel_at = physical.find("BatchKernel lane=Device").unwrap();
        assert!(transfer_at < kernel_at, "{physical}");
        assert_eq!(physical.matches("Transfer(H2D)").count(), 1, "{physical}");

        let cuda_batch = cuda_loader.next_batch().unwrap().unwrap();
        assert_eq!(cuda_batch.images.dims(), [1, 3, 1, 1]);
        assert_eq!(cuda_batch.images.dtype(), DType::F32);
        assert_eq!(cuda_batch.axis_order, ImageAxisOrder::Chw);
        assert!(cuda_batch.images.device().is_cuda());
        if let Device::Cuda(device) = cuda_batch.images.device() {
            let stats = device.debug_stats();
            assert_eq!(stats.kernel_launch_count, 1, "{stats:?}");
            assert_eq!(
                stats.h2d_count, 4,
                "image + label + scale + bias: {stats:?}"
            );
        }
        let cpu_values = cpu_batch.images.to_vec::<f32>().unwrap();
        let cuda_values = cuda_batch.images.to_vec::<f32>().unwrap();
        for (actual, expected) in cuda_values.iter().zip(cpu_values) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
        assert_eq!(cuda_batch.labels.to_vec::<i64>().unwrap(), [7]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sink_fuses_semantic_flip_crop_resize_and_normalize_for_the_batch() {
        let Ok(_device) = Device::cuda(0) else {
            eprintln!("skipping CUDA augmentation test because device 0 is unavailable");
            return;
        };
        let fallback = decoded_stub()
            .random_horizontal_flip(1.0)
            .random_resized_crop(2, 2)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(1, false)
            .cuda_sink(0)
            .compile()
            .unwrap();
        assert_eq!(fallback.plan.sample_op_count(), 2);
        assert_eq!(fallback.plan.first_batch_op_name(), Some("NormalizeToChw"));

        let mean = vec![0.5, 0.25, 0.0];
        let std = vec![0.5, 0.5, 1.0];
        let pipeline = ImagePipeline::from_source(spatial_dense_source())
            .random_horizontal_flip(1.0)
            .random_resized_crop(2, 2)
            .normalize(mean, std)
            .hwc_to_chw()
            .prefetch_batches(0)
            .batch(2, false);

        let mut cpu_loader = pipeline.clone().compile().unwrap();
        let cpu_batch = cpu_loader.next_batch().unwrap().unwrap();
        let mut cuda_loader = pipeline.cuda_sink(0).compile().unwrap();
        assert_eq!(cuda_loader.plan.sample_op_count(), 0);
        assert_eq!(
            cuda_loader.plan.first_batch_op_name(),
            Some("VisionAugmentNormalizeToChw")
        );
        let explanation = cuda_loader.physical_explain().unwrap();
        assert_eq!(
            explanation.matches("Transfer(H2D)").count(),
            1,
            "{explanation}"
        );

        let cuda_batch = cuda_loader.next_batch().unwrap().unwrap();
        assert_eq!(cuda_batch.images.dims(), [2, 3, 2, 2]);
        assert_eq!(cuda_batch.images.dtype(), DType::F32);
        assert_eq!(cuda_batch.axis_order, ImageAxisOrder::Chw);
        assert!(cuda_batch.images.device().is_cuda());
        if let Device::Cuda(device) = cuda_batch.images.device() {
            let stats = device.debug_stats();
            assert_eq!(stats.kernel_launch_count, 1, "{stats:?}");
            assert_eq!(
                stats.h2d_count, 5,
                "image + labels + semantic params + scale + bias: {stats:?}"
            );
        }
        let cpu_values = cpu_batch.images.to_vec::<f32>().unwrap();
        let cuda_values = cuda_batch.images.to_vec::<f32>().unwrap();
        assert_eq!(cpu_values.len(), cuda_values.len());
        for (actual, expected) in cuda_values.iter().zip(cpu_values) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
        assert_eq!(cuda_batch.labels.to_vec::<i64>().unwrap(), [0, 1]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_random_parameters_match_cpu_across_workers_and_shuffled_sample_indices() {
        let Ok(_device) = Device::cuda(0) else {
            eprintln!("skipping CUDA random parity test because device 0 is unavailable");
            return;
        };
        let source = spatial_dense_source();
        let configure = |workers, interpolation| {
            ImagePipeline::from_source(source.clone())
                .resize(4, 4)
                .random_horizontal_flip(0.5)
                .random_resized_crop_with_config(
                    RandomResizedCropConfig::new(3, 2).with_interpolation(interpolation),
                )
                .horizontal_flip()
                .vertical_flip()
                .normalize(vec![0.2, 0.3, 0.4], vec![0.7, 0.8, 0.9])
                .hwc_to_chw()
                .shuffle(71)
                .seed(0xB47C_09D1)
                .epoch(6)
                .workers(workers)
                .batch(2, false)
        };
        let collect = |mut loader: crate::runtime::ImageDataLoader| {
            let mut batches = Vec::new();
            while let Some(batch) = loader.next_batch().unwrap() {
                batches.push(batch);
            }
            batches
        };
        for interpolation in [
            InterpolationMode::Nearest,
            InterpolationMode::Bilinear,
            InterpolationMode::Bicubic,
            InterpolationMode::Lanczos3,
        ] {
            let inline = collect(configure(0, interpolation).compile().unwrap());
            let worker_cpu = collect(configure(3, interpolation).compile().unwrap());
            let cuda = collect(configure(3, interpolation).cuda_sink(0).compile().unwrap());
            assert_eq!(inline.len(), worker_cpu.len());
            assert_eq!(inline.len(), cuda.len());
            for ((reference, worker), device) in inline.iter().zip(&worker_cpu).zip(&cuda) {
                assert_eq!(
                    reference.labels.to_vec::<i64>().unwrap(),
                    worker.labels.to_vec::<i64>().unwrap()
                );
                assert_eq!(
                    reference.labels.to_vec::<i64>().unwrap(),
                    device.labels.to_vec::<i64>().unwrap()
                );
                assert_eq!(&reference.images.dims()[1..], [3, 2, 3]);
                let expected = reference.images.to_vec::<f32>().unwrap();
                assert_eq!(expected, worker.images.to_vec::<f32>().unwrap());
                let actual = device.images.to_vec::<f32>().unwrap();
                assert_eq!(expected.len(), actual.len());
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() <= 1e-6,
                        "{interpolation:?}, labels={:?}, offset={index}, CUDA={actual}, CPU={expected}",
                        reference.labels.to_vec::<i64>().unwrap()
                    );
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sink_ordinal_round_trips_as_plain_logical_runtime_metadata() {
        let pipeline = decoded_stub().batch(1, false).cuda_sink(2);
        let restored = ImagePipeline::from_logical_plan(&pipeline.to_logical_plan()).unwrap();
        assert_eq!(restored.runtime.sink_device_ordinal, Some(2));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_sink_request_without_feature_is_an_explicit_error() {
        let mut pipeline = decoded_stub().batch(1, false);
        pipeline.runtime.sink_device_ordinal = Some(0);
        let error = compile_err(pipeline);
        assert!(error.contains("without the cuda feature"));
    }

    #[test]
    fn logical_round_trip_preserves_compile_errors() {
        let invalid = stub(10).resize(8, 8).batch(4, false);
        let legacy_error = legacy_compile_err(invalid.clone());
        let logical_error = compile_err(invalid);
        assert!(logical_error.contains("node %1"), "got: {logical_error}");
        assert!(
            logical_error.contains(
                legacy_error
                    .strip_prefix("invalid pipeline: ")
                    .unwrap_or(&legacy_error)
            ),
            "legacy error: {legacy_error}; logical error: {logical_error}"
        );
    }

    #[test]
    fn logical_round_trip_preserves_compiled_batch_output() {
        let original = decoded_stub()
            .skip(0)
            .take(1)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(1, false);
        let lowered = ImagePipeline::from_logical_plan(&original.to_logical_plan()).unwrap();
        let mut original_loader = original.compile_legacy_for_test(0).unwrap();
        let mut lowered_loader = lowered.compile().unwrap();
        let original_batch = original_loader.next_batch().unwrap().unwrap();
        let lowered_batch = lowered_loader.next_batch().unwrap().unwrap();

        assert_eq!(
            original_batch.labels.to_vec::<i64>().unwrap(),
            lowered_batch.labels.to_vec::<i64>().unwrap()
        );
        assert_eq!(original_batch.images.dims(), lowered_batch.images.dims());
        assert_eq!(
            original_batch.images.to_vec::<f32>().unwrap(),
            lowered_batch.images.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn property_inference_tracks_resize_layout_and_batch_properties() {
        use rivet_plan::{
            AxisOrder, Contiguity, DataType as PlanDType, NodeKind, Representation, Residency,
            ShapeDim, ValueGranularity,
        };

        let pipeline = stub(4)
            .decode_image()
            .resize(12, 8)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, true);
        let logical = pipeline.to_logical_plan();
        let annotations = logical
            .infer_properties(&super::inference::VisionPropertyInference::new(0))
            .unwrap();
        let node_properties = logical
            .nodes()
            .map(|(id, node)| (node.kind(), annotations.get(id).unwrap()))
            .collect::<Vec<_>>();

        let resized = node_properties
            .iter()
            .find(|(kind, properties)| {
                *kind == NodeKind::Op
                    && properties.shape.as_ref().is_some_and(|shape| {
                        shape.dims() == [ShapeDim::Known(8), ShapeDim::Known(12), ShapeDim::Dynamic]
                    })
            })
            .unwrap()
            .1;
        assert_eq!(resized.representation, Some(Representation::Image));
        assert_eq!(resized.dtype, Some(PlanDType::U8));
        assert_eq!(resized.axis_order, Some(AxisOrder::Hwc));
        assert_eq!(resized.residency, Some(Residency::Host));
        assert_eq!(resized.granularity, Some(ValueGranularity::Sample));
        assert_eq!(resized.contiguity, Some(Contiguity::Contiguous));

        let batch = node_properties
            .iter()
            .find(|(kind, properties)| {
                *kind == NodeKind::Batch && properties.granularity == Some(ValueGranularity::Batch)
            })
            .unwrap()
            .1;
        assert_eq!(batch.dtype, Some(PlanDType::F32));
        assert_eq!(batch.axis_order, Some(AxisOrder::Nchw));
        assert_eq!(
            batch.shape.as_ref().unwrap().dims(),
            [
                ShapeDim::Known(4),
                ShapeDim::Dynamic,
                ShapeDim::Known(8),
                ShapeDim::Known(12)
            ]
        );
        assert_eq!(batch.contiguity, Some(Contiguity::Contiguous));
    }

    #[test]
    fn property_inference_rejects_known_crop_bounds_before_execution() {
        let pipeline = stub(1)
            .decode_image()
            .resize(10, 8)
            .crop(9, 0, 2, 2)
            .batch(1, false);
        let error = pipeline.infer_properties().unwrap_err().to_string();
        assert!(
            error.contains("property inference failed at node %3"),
            "got: {error}"
        );
        assert!(
            error.contains("crop rectangle (9, 0, 2, 2) exceeds image shape 10x8"),
            "got: {error}"
        );
    }

    #[test]
    fn worker_normalize_keeps_sample_stage_property() {
        use rivet_plan::{NodeKind, OperatorStage};

        let pipeline = stub(1)
            .decode_image()
            .horizontal_flip()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .random_erasing(0.0)
            .workers(2)
            .batch(1, false);
        let logical = pipeline.to_logical_plan();
        let annotations = pipeline.infer_properties().unwrap();
        let last_op = logical
            .nodes()
            .filter(|(_, node)| node.kind() == NodeKind::Op)
            .last()
            .unwrap()
            .0;
        assert_eq!(
            annotations.get(last_op).unwrap().operator.unwrap().stage,
            OperatorStage::Sample
        );
    }

    #[test]
    fn property_inference_covers_every_image_op_variant() {
        use crate::pipeline::op::ImageOp;
        use crate::transforms::{Point2, RotationAngle};
        use rivet_core::DType;

        let points = [
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let operations = vec![
            ImageOp::resize(4, 4),
            ImageOp::crop(0, 0, 1, 1),
            ImageOp::center_crop(1, 1),
            ImageOp::pad(1),
            ImageOp::horizontal_flip(),
            ImageOp::vertical_flip(),
            ImageOp::random_crop(1, 1, 0),
            ImageOp::random_resized_crop(1, 1),
            ImageOp::random_horizontal_flip(0.5),
            ImageOp::brightness(1),
            ImageOp::contrast(1.0),
            ImageOp::hue(1),
            ImageOp::color_jitter(1, 1.0, 1),
            ImageOp::invert(),
            ImageOp::posterize(4),
            ImageOp::solarize(128),
            ImageOp::autocontrast(),
            ImageOp::equalize(),
            ImageOp::sharpness(1.0),
            ImageOp::arbitrary_rotate(0.0),
            ImageOp::random_affine(0.0),
            ImageOp::perspective(points, points),
            ImageOp::random_perspective(0.1, 0.5),
            ImageOp::elastic_transform(1.0, 1.0),
            ImageOp::random_apply(0.5, vec![ImageOp::invert()]),
            ImageOp::random_choice(vec![vec![ImageOp::invert()], vec![ImageOp::solarize(128)]]),
            ImageOp::random_order(vec![ImageOp::invert(), ImageOp::solarize(128)]),
            ImageOp::gaussian_blur(1.0),
            ImageOp::grayscale(3),
            ImageOp::random_grayscale(0.5, 3),
            ImageOp::random_erasing(0.0),
            ImageOp::convert_image_dtype(DType::F32),
            ImageOp::rotate(RotationAngle::Deg90),
            ImageOp::normalize(vec![0.5; 3], vec![0.5; 3]),
            ImageOp::hwc_to_chw(),
        ];

        for op in operations {
            let mut pipeline = decoded_stub();
            pipeline.ops.push(op.clone());
            pipeline.batch = Some(crate::pipeline::op::BatchConfig::new(1, false));
            pipeline
                .infer_properties()
                .unwrap_or_else(|error| panic!("{} inference failed: {error}", op.name()));
        }

        let mut encoded_pipeline = stub(1);
        encoded_pipeline.ops.push(ImageOp::decode());
        encoded_pipeline.batch = Some(crate::pipeline::op::BatchConfig::new(1, false));
        encoded_pipeline.infer_properties().unwrap();
    }
}
