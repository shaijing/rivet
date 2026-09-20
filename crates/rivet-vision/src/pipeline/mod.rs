mod builder;
mod compile;
pub mod op;
mod transform;

pub use builder::ImagePipeline;
pub use transform::{Compose, ImageTransform, TransformSequence};

#[cfg(test)]
mod tests {
    use super::{ImagePipeline, TransformSequence};
    use crate::cache::DenseImageMemoryDataset;
    use crate::pipeline::op::{ExecutionKind, PipelineImageState};
    use crate::sample::image::{DecodedSample, EncodedImageSample, ImageAxisOrder};
    use crate::source::ImageSource;
    use crate::transforms::Point2;
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

    fn compile_err(pipeline: ImagePipeline) -> String {
        match pipeline.compile() {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected a compile error"),
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
}
