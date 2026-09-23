use crate::batch::ImageBatchBuilder;
use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::sample::image::{DecodedSample, ImageBatch, ImageSample};
use crate::sampler::IndexSampler;
use rivet_exec::physical::PhysicalGraph;
use rivet_exec::runtime::{PhysicalPipelineAdapter, PhysicalPipelineExecutor, PipelineError};
use std::sync::Arc;

struct ImagePipelineAdapter {
    plan: Arc<ExecutionPlan>,
}

impl PhysicalPipelineAdapter for ImagePipelineAdapter {
    type Sample = ImageSample;
    type Output = DecodedSample;
    type Batch = ImageBatch;
    type BatchBuilder = ImageBatchBuilder;
    type Error = RivetError;

    fn batch_size(&self) -> usize {
        self.plan.batch.size
    }

    fn drop_last(&self) -> bool {
        self.plan.batch.drop_last
    }

    fn is_batch_native(&self) -> bool {
        self.plan.can_use_batch_native()
    }

    fn fetch_samples(&self, indices: &[usize]) -> Result<Vec<ImageSample>, RivetError> {
        self.plan.source.get_many(indices)
    }

    fn process_sample(
        &self,
        sample: ImageSample,
        sample_index: usize,
    ) -> Result<DecodedSample, RivetError> {
        self.plan.apply_sample_ops(sample, sample_index)
    }

    fn worker_panic_error(&self, worker_id: usize, sample_index: usize) -> RivetError {
        RivetError::Worker(format!(
            "worker {worker_id} panicked while processing sample {sample_index}"
        ))
    }

    fn sample_bytes(&self, sample: &ImageSample) -> usize {
        match sample {
            ImageSample::Encoded(sample) => sample.image.len(),
            ImageSample::Decoded(sample) => sample.image.logical_bytes(),
        }
    }

    fn output_bytes(&self, output: &DecodedSample) -> usize {
        output.image.logical_bytes()
    }

    fn batch_bytes(&self, batch: &ImageBatch) -> usize {
        batch
            .images
            .logical_bytes()
            .saturating_add(batch.labels.logical_bytes())
    }

    fn fetch_batch(&self, indices: &[usize]) -> Result<Option<ImageBatch>, RivetError> {
        self.plan.source.get_batch(indices).transpose()
    }

    fn batch_builder(&self, capacity: usize) -> ImageBatchBuilder {
        ImageBatchBuilder::with_capacity(capacity)
    }

    fn push_batch_sample(
        &self,
        builder: &mut ImageBatchBuilder,
        sample: DecodedSample,
    ) -> Result<(), RivetError> {
        builder.push(sample)
    }

    fn finish_batch(&self, builder: ImageBatchBuilder) -> Result<ImageBatch, RivetError> {
        builder.finish()
    }

    fn apply_batch(&self, batch: ImageBatch) -> Result<ImageBatch, RivetError> {
        self.plan.apply_batch_ops(batch)
    }
}

pub struct ImageDataLoader {
    pub plan: Arc<ExecutionPlan>,
    pub sampler: IndexSampler,
    executor: PhysicalPipelineExecutor<ImagePipelineAdapter>,
}

impl ImageDataLoader {
    pub(crate) fn new(
        plan: Arc<ExecutionPlan>,
        sampler: IndexSampler,
        num_workers: usize,
        prefetch_batches: usize,
        physical: PhysicalGraph,
    ) -> RivetResult<Self> {
        let adapter = Arc::new(ImagePipelineAdapter {
            plan: Arc::clone(&plan),
        });
        let executor =
            PhysicalPipelineExecutor::with_graph(adapter, num_workers, prefetch_batches, physical)
                .map_err(|error| RivetError::Worker(error.to_string()))?;

        Ok(Self {
            plan,
            sampler,
            executor,
        })
    }

    pub fn next_batch(&mut self) -> RivetResult<Option<ImageBatch>> {
        self.executor
            .next_batch(&mut self.sampler)
            .map_err(|error| match error {
                PipelineError::Runtime(error) => RivetError::Worker(error.to_string()),
                PipelineError::Domain(error) => error,
            })
    }

    /// Deterministic physical stage graph selected by this loader.
    pub fn physical_explain(&mut self) -> RivetResult<String> {
        self.executor
            .physical_explain()
            .map_err(|error| RivetError::Worker(error.to_string()))
    }
    /// Convenience wrapper for `(&mut self).into_iter()`.
    pub fn iter(&mut self) -> ImageDataLoaderIter<'_> {
        ImageDataLoaderIter {
            loader: self,
            done: false,
        }
    }
}

/// Fallible iterator over an [`ImageDataLoader`]'s batches.
///
/// Item is `RivetResult<ImageBatch>` so errors are reported once and then
/// the iterator terminates: after EOF or a single error, every further
/// `next()` returns `None`. This matches the loader's terminal failed-state
/// model instead of yielding endless error items.
pub struct ImageDataLoaderIter<'a> {
    loader: &'a mut ImageDataLoader,
    done: bool,
}

impl Iterator for ImageDataLoaderIter<'_> {
    type Item = RivetResult<ImageBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        match self.loader.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                Some(Err(err))
            }
        }
    }
}

impl<'a> IntoIterator for &'a mut ImageDataLoader {
    type Item = RivetResult<ImageBatch>;
    type IntoIter = ImageDataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ImageDataLoaderIter {
            loader: self,
            done: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::DenseImageMemoryDataset;
    use crate::pipeline::{ImagePipeline, TransformSequence};
    use crate::sample::image::EncodedImageSample;
    use crate::source::ImageSource;
    use arrow_buffer::Buffer;
    use rivet_core::{DType, Device, Tensor};
    use rivet_data::dataset::Dataset;
    use rivet_data::random::OpKey;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // 1x1 RGB PNG (red pixel), valid input for decode_image.
    const PNG_1X1: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

    struct SampleDataset {
        len: usize,
        err_at: Option<usize>,
        panic_at: Option<usize>,
        /// Simulated per-sample latency for rows 0..8.
        slow_first_batch_ms: u64,
    }

    impl Dataset for SampleDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.len
        }

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            indices.iter().map(|&index| self.get_one(index)).collect()
        }
    }

    impl SampleDataset {
        fn get_one(&self, index: usize) -> rivet_data::DataResult<EncodedImageSample> {
            if self.err_at == Some(index) {
                return Err(rivet_data::errors::invalid_argument("boom"));
            }
            if self.panic_at == Some(index) {
                panic!("sample {index} panics");
            }
            if self.slow_first_batch_ms > 0 && index < 8 {
                std::thread::sleep(std::time::Duration::from_millis(self.slow_first_batch_ms));
            }
            Ok(EncodedImageSample {
                image: Buffer::from(PNG_1X1.to_vec()),
                label: index as i64,
            })
        }
    }

    struct CountingDataset {
        len: usize,
        calls: Arc<AtomicUsize>,
    }

    impl Dataset for CountingDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.len
        }

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            indices
                .iter()
                .map(|&index| {
                    Ok(EncodedImageSample {
                        image: Buffer::from(PNG_1X1.to_vec()),
                        label: index as i64,
                    })
                })
                .collect()
        }
    }

    fn pipeline(len: usize, workers: usize) -> ImagePipeline {
        ImagePipeline::new(Arc::new(SampleDataset {
            len,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 0,
        }))
        .decode_image()
        .workers(workers)
    }

    #[test]
    fn source_fetch_scales_with_logical_batches() {
        for workers in [0, 3] {
            let calls = Arc::new(AtomicUsize::new(0));
            let dataset = Arc::new(CountingDataset {
                len: 32,
                calls: Arc::clone(&calls),
            });
            let mut loader = ImagePipeline::new(dataset)
                .decode_image()
                .workers(workers)
                .batch(8, false)
                .compile()
                .unwrap();

            let batches = drain(&mut loader);
            assert_eq!(batches.len(), 4);
            assert_eq!(calls.load(Ordering::Relaxed), 4, "workers={workers}");
        }
    }

    #[test]
    fn image_loader_exposes_logical_to_physical_stage_lowering() {
        let mut loader = pipeline(4, 0)
            .normalize(vec![0.0; 3], vec![1.0; 3])
            .batch(2, false)
            .compile()
            .unwrap();
        let physical = loader.physical_explain().unwrap();
        assert!(physical.contains("PhysicalGraph"));
        assert!(physical.contains("Sampler lane=Cpu"));
        assert!(physical.contains("Source lane=Io"));
        assert!(physical.contains("SampleKernel lane=Cpu"));
        assert!(physical.contains("Batch lane=Cpu"));
        assert!(physical.contains("BatchKernel lane=Cpu"));
        assert!(physical.contains("Sink lane=Cpu"));
        assert!(
            physical.find("Batch lane=Cpu").unwrap()
                < physical.find("BatchKernel lane=Cpu").unwrap()
        );
        assert!(
            physical.find("Sampler lane=Cpu").unwrap() < physical.find("Source lane=Io").unwrap()
        );

        assert_eq!(drain(&mut loader).len(), 2);
        assert!(
            loader
                .executor
                .profiler()
                .snapshot()
                .values()
                .any(|entry| { entry.executions > 0 && entry.elapsed.as_nanos() > 0 })
        );
    }

    #[test]
    fn physical_runtime_matches_direct_execution_plan_batches() {
        let mut loader = pipeline(9, 3)
            .resize(1, 1)
            .batch(4, false)
            .compile()
            .unwrap();
        let plan = Arc::clone(&loader.plan);
        let physical = drain(&mut loader);

        let mut direct = Vec::new();
        for start in (0..plan.source.len()).step_by(plan.batch.size) {
            let end = start.saturating_add(plan.batch.size).min(plan.source.len());
            let indices = (start..end).collect::<Vec<_>>();
            let samples = plan.source.get_many(&indices).unwrap();
            let mut builder = ImageBatchBuilder::with_capacity(indices.len());
            for (index, sample) in indices.into_iter().zip(samples) {
                builder
                    .push(plan.apply_sample_ops(sample, index).unwrap())
                    .unwrap();
            }
            direct.push(plan.apply_batch_ops(builder.finish().unwrap()).unwrap());
        }

        assert_batches_equal(&physical, &direct);
    }

    fn drain(loader: &mut ImageDataLoader) -> Vec<ImageBatch> {
        loader.into_iter().collect::<RivetResult<Vec<_>>>().unwrap()
    }

    fn assert_batches_equal(left: &[ImageBatch], right: &[ImageBatch]) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right) {
            assert_eq!(labels(a), labels(b));
            assert_eq!(a.images.dims(), b.images.dims());
            assert_eq!(a.images.dtype(), b.images.dtype());
            match a.images.dtype() {
                DType::U8 => assert_eq!(
                    a.images.to_vec::<u8>().unwrap(),
                    b.images.to_vec::<u8>().unwrap()
                ),
                DType::F32 => assert_eq!(
                    a.images.to_vec::<f32>().unwrap(),
                    b.images.to_vec::<f32>().unwrap()
                ),
                dtype => panic!("unexpected test dtype: {dtype:?}"),
            }
        }
    }

    fn labels(batch: &ImageBatch) -> Vec<i64> {
        batch.labels.to_vec::<i64>().unwrap()
    }

    fn decoded_pipeline(workers: usize) -> ImagePipeline {
        const SAMPLES: usize = 24;
        const HEIGHT: usize = 12;
        const WIDTH: usize = 12;
        const CHANNELS: usize = 3;
        let values = (0..SAMPLES * HEIGHT * WIDTH * CHANNELS)
            .map(|index| ((index * 17 + index / 11) % 251) as u8)
            .collect::<Vec<_>>();
        let images =
            Tensor::from_vec(values, [SAMPLES, HEIGHT, WIDTH, CHANNELS], &Device::Cpu).unwrap();
        let labels = Tensor::from_vec(
            (0..SAMPLES).map(|index| index as i64).collect::<Vec<_>>(),
            [SAMPLES],
            &Device::Cpu,
        )
        .unwrap();
        let dataset = DenseImageMemoryDataset::new(images, labels).unwrap();
        ImagePipeline::from_source(ImageSource::from_dense_decoded(Arc::new(dataset)))
            .workers(workers)
    }

    fn collect_decoded(loader: &mut ImageDataLoader) -> Vec<ImageBatch> {
        drain(loader)
    }

    fn assert_random_op_is_worker_independent(
        name: &str,
        build: impl Fn(ImagePipeline) -> ImagePipeline,
    ) {
        let collect = |workers| {
            let mut loader = build(decoded_pipeline(workers))
                .seed(0x51A7)
                .epoch(7)
                .batch(4, false)
                .prefetch_batches(2)
                .compile()
                .unwrap();
            collect_decoded(&mut loader)
        };

        let expected = collect(0);
        for workers in [1, 4, 8] {
            assert_batches_equal(&expected, &collect(workers));
            assert_eq!(expected.len(), 6, "{name} should produce all batches");
        }
    }

    #[test]
    fn workers_match_inline_order_and_content() {
        let mut inline = pipeline(37, 0).batch(8, false).compile().unwrap();
        let mut pooled = pipeline(37, 4).batch(8, false).compile().unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        assert_eq!(inline.iter().map(|b| b.labels.dims()[0]).sum::<usize>(), 37);
    }

    #[test]
    fn workers_match_inline_for_batch_stage() {
        let mut inline = pipeline(17, 0)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false)
            .compile()
            .unwrap();
        let mut pooled = pipeline(17, 3)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false)
            .compile()
            .unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        assert_eq!(inline[0].images.dims()[1], 3);
        assert_eq!(inline[0].images.dtype(), rivet_core::DType::F32);
    }

    #[test]
    fn phase2_random_ops_match_inline_and_workers() {
        let configure = |workers| {
            pipeline(19, workers)
                .pad(1)
                .random_resized_crop(1, 1)
                .color_jitter(4, 0.2, 10)
                .random_grayscale(0.5, 3)
                .random_erasing(0.5)
                .shuffle(17)
                .batch(4, false)
        };
        let mut inline = configure(0).compile().unwrap();
        let mut pooled = configure(4).prefetch_batches(2).compile().unwrap();
        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
    }

    #[test]
    fn phase4_control_ops_match_inline_and_workers() {
        let configure = |workers| {
            pipeline(19, workers)
                .epoch(4)
                .random_apply(0.5, TransformSequence::new().brightness(10))
                .random_choice(vec![
                    TransformSequence::new().invert(),
                    TransformSequence::new().brightness(3),
                ])
                .random_order(TransformSequence::new().contrast(0.8).invert())
                .shuffle(23)
                .batch(4, false)
        };
        let mut inline = configure(0).compile().unwrap();
        let mut pooled = configure(4).prefetch_batches(2).compile().unwrap();
        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
    }

    #[test]
    fn phase5_random_geometry_matches_inline_and_workers() {
        let configure = |workers| {
            pipeline(19, workers)
                .epoch(4)
                .random_affine(12.0)
                .elastic_transform(1.0, 1.0)
                .shuffle(23)
                .batch(4, false)
        };
        let mut inline = configure(0).compile().unwrap();
        let mut pooled = configure(4).prefetch_batches(2).compile().unwrap();
        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
    }

    #[test]
    fn each_random_transform_is_deterministic_across_worker_counts() {
        assert_random_op_is_worker_independent("RandomCrop", |pipeline| {
            pipeline.random_crop(8, 8, 2)
        });
        assert_random_op_is_worker_independent("RandomResizedCrop", |pipeline| {
            pipeline.random_resized_crop(8, 8)
        });
        assert_random_op_is_worker_independent("RandomHorizontalFlip", |pipeline| {
            pipeline.random_horizontal_flip(0.5)
        });
        assert_random_op_is_worker_independent("ColorJitter", |pipeline| {
            pipeline.color_jitter(24, 0.7, 24)
        });
        assert_random_op_is_worker_independent("RandomErasing", |pipeline| {
            pipeline.random_erasing(0.75)
        });
    }

    #[test]
    fn same_seed_epoch_repeats_and_new_epoch_changes_random_output() {
        let collect = |epoch| {
            let mut loader = decoded_pipeline(4)
                .random_crop(8, 8, 2)
                .random_horizontal_flip(0.5)
                .color_jitter(24, 0.7, 24)
                .random_erasing(0.75)
                .seed(0xD15EA5E)
                .epoch(epoch)
                .batch(4, false)
                .compile()
                .unwrap();
            collect_decoded(&mut loader)
        };

        let first = collect(0);
        assert_batches_equal(&first, &collect(0));
        assert_ne!(
            first[0].images.to_vec::<u8>().unwrap(),
            collect(1)[0].images.to_vec::<u8>().unwrap(),
            "a new epoch must change stochastic output"
        );
    }

    #[test]
    fn nested_random_control_ops_are_deterministic_and_worker_independent() {
        let configure = |workers, epoch| {
            decoded_pipeline(workers)
                .random_apply(
                    0.75,
                    TransformSequence::new()
                        .random_horizontal_flip(0.5)
                        .color_jitter(18, 0.5, 12),
                )
                .random_choice(vec![
                    TransformSequence::new().random_crop(8, 8, 0),
                    TransformSequence::new().random_crop(8, 8, 1),
                ])
                .random_order(TransformSequence::new().brightness(11).contrast(0.7))
                .seed(0xA11CE)
                .epoch(epoch)
                .batch(4, false)
                .prefetch_batches(3)
        };

        let collect = |workers, epoch| {
            let mut loader = configure(workers, epoch).compile().unwrap();
            collect_decoded(&mut loader)
        };
        let inline = collect(0, 4);
        for workers in [1, 4, 8] {
            assert_batches_equal(&inline, &collect(workers, 4));
        }
        assert_batches_equal(&inline, &collect(0, 4));
        assert_ne!(
            inline[0].images.to_vec::<u8>().unwrap(),
            collect(0, 5)[0].images.to_vec::<u8>().unwrap()
        );
    }

    #[test]
    fn duplicate_random_transforms_get_distinct_namespaces() {
        let mut loader = decoded_pipeline(0)
            .random_crop(10, 10, 0)
            .random_crop(8, 8, 0)
            .seed(91)
            .batch(4, false)
            .compile()
            .unwrap();
        let keys = loader
            .plan
            .sample_ops
            .iter()
            .map(|op| op.random_key.expect("random crop key"))
            .collect::<Vec<_>>();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], OpKey::from_parts("RandomCrop", 0));
        assert_eq!(keys[1], OpKey::from_parts("RandomCrop", 1));
        assert_eq!(collect_decoded(&mut loader)[0].images.dims(), &[4, 8, 8, 3]);
    }

    #[test]
    fn workers_preserve_skip_take_and_drop_last() {
        let mut inline = pipeline(50, 0)
            .skip(2)
            .take(11)
            .batch(5, true)
            .compile()
            .unwrap();
        let mut pooled = pipeline(50, 3)
            .skip(2)
            .take(11)
            .batch(5, true)
            .compile()
            .unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        // 11 taken rows, drop_last keeps 5+5, second batch is the last.
        assert_eq!(inline.len(), 2);
        assert_eq!(labels(&inline[0]), vec![2, 3, 4, 5, 6]);
        assert_eq!(labels(&inline[1]), vec![7, 8, 9, 10, 11]);
    }

    #[test]
    fn sample_error_propagates_from_workers() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: Some(1),
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let err = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected a worker error"),
        };
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    #[test]
    fn worker_panic_becomes_error_not_hang() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: None,
            panic_at: Some(3),
            slow_first_batch_ms: 0,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let err = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected a worker panic error"),
        };
        assert!(err.to_string().contains("panicked"), "got: {err}");
    }

    #[test]
    fn prefetch_configs_preserve_sampler_order() {
        let mut baseline = pipeline(101, 0)
            .skip(3)
            .take(93)
            .batch(10, true)
            .compile()
            .unwrap();
        let baseline = drain(&mut baseline);

        for (workers, prefetch) in [(1, 0), (2, 1), (2, 3), (4, 4), (8, 2)] {
            let mut loader = pipeline(101, workers)
                .skip(3)
                .take(93)
                .batch(10, true)
                .prefetch_batches(prefetch)
                .compile()
                .unwrap();
            let pooled = drain(&mut loader);
            assert_batches_equal(&baseline, &pooled);
        }
    }

    #[test]
    fn reordered_worker_completion_preserves_sampler_order() {
        let mut inline = ImagePipeline::new(Arc::new(SampleDataset {
            len: 24,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 5,
        }))
        .decode_image()
        .shuffle(77)
        .batch(4, false)
        .compile()
        .unwrap();
        let expected = drain(&mut inline);

        let mut pooled = ImagePipeline::new(Arc::new(SampleDataset {
            len: 24,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 5,
        }))
        .decode_image()
        .shuffle(77)
        .workers(4)
        .prefetch_batches(4)
        .batch(4, false)
        .compile()
        .unwrap();
        assert_batches_equal(&expected, &drain(&mut pooled));
    }

    #[test]
    fn prefetch_stops_after_dropped_tail() {
        // 30..43 spans 13 taken rows; drop_last with size 5 yields 5+5 and
        // silently discards the 3-row tail even when batches run ahead.
        for (workers, prefetch) in [(2, 3), (4, 1)] {
            let mut loader = pipeline(50, workers)
                .skip(30)
                .take(13)
                .batch(5, true)
                .prefetch_batches(prefetch)
                .compile()
                .unwrap();
            let batches = drain(&mut loader);
            assert_eq!(batches.len(), 2, "workers={workers} prefetch={prefetch}");
            assert_eq!(labels(&batches[0]), vec![30, 31, 32, 33, 34]);
            assert_eq!(labels(&batches[1]), vec![35, 36, 37, 38, 39]);
        }
    }

    #[test]
    fn drop_shuts_down_workers_without_panicking() {
        let mut loader = pipeline(10, 4).batch(4, false).compile().unwrap();
        assert!(loader.next_batch().unwrap().is_some());
        // Dropping the loader joins every worker; nothing should hang.
    }

    /// A sample error must put the loader into a terminal failed state:
    /// the next call errors immediately instead of waiting forever for the
    /// result that will never arrive.
    #[test]
    fn next_batch_after_worker_error_returns_terminal_error() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: Some(1),
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(2)
            .prefetch_batches(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let first = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the sample error"),
        };
        assert!(first.to_string().contains("boom"), "got: {first}");

        let second = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the terminal failed-state error"),
        };
        assert!(second.to_string().contains("failed state"), "got: {second}");
    }

    /// Deep prefetch with early drop must not deadlock: workers blocked on
    /// a full result queue are released by the result-receiver disconnect.
    #[test]
    fn drop_with_full_prefetch_does_not_hang() {
        let mut loader = pipeline(1000, 4)
            .batch(8, false)
            .prefetch_batches(8)
            .compile()
            .unwrap();
        let _ = loader.next_batch().unwrap();
        drop(loader);
    }

    /// Even when batch 1 completes before the slower batch 0, delivery must
    /// return batch 0 first.
    #[test]
    fn slow_earlier_batch_preserves_cross_batch_order() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 30,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(3)
            .prefetch_batches(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let mut seen = Vec::new();
        while let Some(batch) = loader.next_batch().unwrap() {
            seen.extend(labels(&batch));
        }
        assert_eq!(seen, (0..20).collect::<Vec<i64>>());
    }

    /// Extreme batch/prefetch sizes must fail at compile with an argument
    /// error before any channel is constructed (no panic, no allocation).
    #[test]
    fn worker_queue_capacity_overflow_is_rejected() {
        let dataset = Arc::new(SampleDataset {
            len: 10,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let result = ImagePipeline::new(dataset)
            .decode_image()
            .workers(4)
            .prefetch_batches(usize::MAX)
            .batch(usize::MAX, false)
            .compile();
        let err = match result {
            Err(err) => err,
            Ok(_) => panic!("expected a capacity overflow error"),
        };
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }

    /// Inline errors must be terminal too: worker count must not change the
    /// loader's error semantics.
    #[test]
    fn inline_error_is_terminal_too() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: Some(1),
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(0)
            .batch(4, false)
            .compile()
            .unwrap();

        let first = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the sample error"),
        };
        assert!(first.to_string().contains("boom"), "got: {first}");

        let second = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the terminal failed-state error"),
        };
        assert!(second.to_string().contains("failed state"), "got: {second}");
    }

    #[test]
    fn iterator_stops_at_eof() {
        let mut loader = pipeline(10, 0).batch(4, false).compile().unwrap();

        let batches = (&mut loader)
            .into_iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(labels(&batches[0]), vec![0, 1, 2, 3]);

        let mut iter = (&mut loader).into_iter();
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn iterator_yields_error_once_then_stops() {
        for workers in [0, 3] {
            let dataset = Arc::new(SampleDataset {
                len: 20,
                err_at: Some(1),
                panic_at: None,
                slow_first_batch_ms: 0,
            });
            let mut loader = ImagePipeline::new(dataset)
                .decode_image()
                .workers(workers)
                .batch(4, false)
                .compile()
                .unwrap();

            let mut iter = (&mut loader).into_iter();
            let first = iter.next().expect("must yield the failure");
            let err = match first {
                Err(err) => err,
                Ok(_) => panic!("expected an error item"),
            };
            assert!(err.to_string().contains("boom"), "got: {err}");

            // The error is reported exactly once; the iterator then ends.
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn iterator_matches_inline_and_workers() {
        let mut inline = pipeline(37, 0)
            .skip(2)
            .take(30)
            .batch(8, true)
            .compile()
            .unwrap();
        let mut workers = pipeline(37, 3)
            .skip(2)
            .take(30)
            .batch(8, true)
            .prefetch_batches(2)
            .compile()
            .unwrap();

        let inline = drain(&mut inline);
        let workers = drain(&mut workers);
        assert_batches_equal(&inline, &workers);
    }

    #[test]
    fn iter_method_equals_into_iter() {
        let mut via_iter_loader = pipeline(10, 2).batch(4, false).compile().unwrap();
        let mut via_into_loader = pipeline(10, 2).batch(4, false).compile().unwrap();

        let via_iter = via_iter_loader
            .iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        let via_into = (&mut via_into_loader)
            .into_iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        assert_batches_equal(&via_iter, &via_into);
    }

    #[test]
    fn shuffle_is_deterministic_across_workers() {
        let mut loader_a = pipeline(50, 0)
            .shuffle(7)
            .batch(8, false)
            .compile()
            .unwrap();
        let mut loader_b = pipeline(50, 4)
            .shuffle(7)
            .batch(8, false)
            .prefetch_batches(2)
            .compile()
            .unwrap();

        let a = drain(&mut loader_a);
        let b = drain(&mut loader_b);
        assert_batches_equal(&a, &b);

        let mut seen = Vec::new();
        for batch in &a {
            seen.extend(labels(batch));
        }
        assert_eq!(
            seen.iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            50
        );

        // A different seed changes the order.
        let mut loader_c = pipeline(50, 0)
            .shuffle(8)
            .batch(8, false)
            .compile()
            .unwrap();
        let c = drain(&mut loader_c);
        let c_labels: Vec<i64> = c.iter().flat_map(labels).collect();
        assert_ne!(c_labels, seen, "different seeds must reorder");
    }

    #[test]
    fn shuffle_applies_after_skip_and_take() {
        // Window is rows 10..=29 after skip/take; the seed only permutes
        // that window.
        let mut loader = pipeline(40, 0)
            .skip(10)
            .take(20)
            .shuffle(3)
            .batch(5, false)
            .compile()
            .unwrap();
        let batches = drain(&mut loader);
        let mut seen: Vec<i64> = batches.iter().flat_map(labels).collect();
        seen.sort_unstable();
        assert_eq!(seen, (10..30).collect::<Vec<i64>>(), "window preserved");
    }

    #[test]
    fn epoch_changes_sampler_namespace_when_seed_is_pipeline_owned() {
        let collect_labels = |epoch| {
            let mut loader = pipeline(50, 0)
                .seed(7)
                .shuffle(11)
                .epoch(epoch)
                .batch(8, false)
                .compile()
                .unwrap();
            drain(&mut loader)
                .iter()
                .flat_map(labels)
                .collect::<Vec<_>>()
        };

        let first = collect_labels(0);
        assert_eq!(first, collect_labels(0));
        assert_ne!(first, collect_labels(1));
    }
}
