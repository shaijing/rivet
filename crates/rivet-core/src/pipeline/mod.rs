pub mod op;

use crate::dataset::source::{Dataset, Source};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::pipeline::op::{
    BatchConfig, ExecutionPlan, IndexOp, PipelineImageState, ImageOp, SourceOp, compile_sampler,
};
use crate::runtime::{ImageDataLoader, RuntimeConfig};
use crate::sample::image::EncodedImageSample;
use crate::sampler::IndexSampler;
use std::sync::Arc;

#[derive(Clone)]
pub struct ImagePipeline {
    pub source: SourceOp,
    pub index_ops: Vec<IndexOp>,
    pub ops: Vec<ImageOp>,
    pub batch: Option<BatchConfig>,
    pub runtime: RuntimeConfig,
}

impl ImagePipeline {
    pub fn new<T>(dataset: Arc<T>) -> Self
    where
        T: Dataset<Item = EncodedImageSample> + 'static,
    {
        Self::from_source(Source::new(dataset))
    }

    pub fn from_source(source: Source<EncodedImageSample>) -> Self {
        Self {
            source: SourceOp::new(source),
            index_ops: Vec::new(),
            ops: Vec::new(),
            batch: None,
            runtime: RuntimeConfig::default(),
        }
    }

    pub fn decode_image(mut self) -> Self {
        self.ops.push(ImageOp::decode());
        self
    }

    pub fn resize(mut self, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::resize(width, height));
        self
    }

    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::crop(x, y, width, height));
        self
    }

    pub fn center_crop(mut self, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::center_crop(width, height));
        self
    }

    pub fn horizontal_flip(mut self) -> Self {
        self.ops.push(ImageOp::horizontal_flip());
        self
    }

    pub fn vertical_flip(mut self) -> Self {
        self.ops.push(ImageOp::vertical_flip());
        self
    }

    pub fn brightness(mut self, value: i32) -> Self {
        self.ops.push(ImageOp::brightness(value));
        self
    }

    pub fn contrast(mut self, value: f32) -> Self {
        self.ops.push(ImageOp::contrast(value));
        self
    }

    pub fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.ops.push(ImageOp::normalize(mean, std));
        self
    }

    pub fn hwc_to_chw(mut self) -> Self {
        self.ops.push(ImageOp::hwc_to_chw());
        self
    }

    pub fn chw_to_hwc(mut self) -> Self {
        self.ops.push(ImageOp::chw_to_hwc());
        self
    }

    pub fn skip(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Skip { count });
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Take { count });
        self
    }

    pub fn batch(mut self, size: usize, drop_last: bool) -> Self {
        self.batch = Some(BatchConfig::new(size, drop_last));
        self
    }

    /// Execute sample loading on a persistent pool of `num_workers` threads
    /// (`0` keeps the synchronous inline path). Ordering, batching and
    /// sampling semantics are unaffected by the worker count.
    pub fn workers(mut self, num_workers: usize) -> Self {
        self.runtime.num_workers = num_workers;
        self
    }

    /// Prepare up to `prefetch_batches` future batches while the caller
    /// consumes the current one (worker pools only; the current batch is
    /// always in flight, so total in-flight = `prefetch_batches + 1`).
    /// Delivery stays in sampler order.
    pub fn prefetch_batches(mut self, prefetch_batches: usize) -> Self {
        self.runtime.prefetch_batches = prefetch_batches;
        self
    }

    pub fn compile(self) -> RivetResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub fn compile_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        let output_state = validate_image_ops(&self.ops)?;

        let batch = self
            .batch
            .ok_or_else(|| invalid_pipeline("pipeline requires .batch(size)"))?;
        batch.validate()?;

        let len = self.source.len();
        let sampler = compile_sampler(len, &self.index_ops);
        let plan = ExecutionPlan {
            source: self.source,
            sampler,
            ops: self.ops,
            batch,
            output_state,
        };
        let num_workers = self.runtime.num_workers;
        let prefetch_batches = self.runtime.prefetch_batches;
        let plan = Arc::new(plan);

        ImageDataLoader::new(
            Arc::clone(&plan),
            IndexSampler::new(plan.sampler.clone(), start),
            num_workers,
            prefetch_batches,
        )
    }
}

fn validate_image_ops(ops: &[ImageOp]) -> RivetResult<PipelineImageState> {
    let mut state = PipelineImageState::Encoded;

    for op in ops {
        op.validate()?;
        state = op.transition(state)?;
    }

    match state {
        PipelineImageState::Encoded => Err(invalid_pipeline(
            "pipeline must decode images before batching",
        )),
        PipelineImageState::Decoded { .. } => Ok(state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::sample::image::{EncodedImageSample, ImageDType, ImageLayout};
    use arrow_buffer::Buffer;

    struct StubDataset {
        len: usize,
    }

    impl Dataset for StubDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.len
        }

        fn get(&self, _index: usize) -> RivetResult<Self::Item> {
            Ok(EncodedImageSample {
                image: Buffer::from(Vec::<u8>::new()),
                label: 0,
            })
        }
    }

    fn stub(len: usize) -> ImagePipeline {
        ImagePipeline::new(Arc::new(StubDataset { len }))
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
        assert!(err.contains("Resize requires a decoded image"), "got: {err}");
    }

    #[test]
    fn normalize_before_decode_rejected_at_compile() {
        let err = compile_err(stub(10).normalize(vec![0.5; 3], vec![0.5; 3]).batch(4, false));
        assert!(err.contains("Normalize requires a decoded image"), "got: {err}");
    }

    #[test]
    fn double_decode_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().decode_image().batch(4, false));
        assert!(err.contains("Decode requires an encoded image"), "got: {err}");
    }

    #[test]
    fn batching_while_encoded_rejected_at_compile() {
        let err = compile_err(stub(10).batch(4, false));
        assert!(err.contains("must decode images before batching"), "got: {err}");
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
    fn invalid_normalize_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().normalize(vec![0.5; 3], vec![0.5; 2]).batch(4, false));
        assert!(err.contains("same length"), "got: {err}");
    }

    #[test]
    fn zero_batch_size_rejected_at_compile() {
        let err = compile_err(stub(10).decode_image().batch(0, false));
        assert!(err.contains("batch size must be greater than 0"), "got: {err}");
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
                dtype: ImageDType::F32,
                layout: ImageLayout::Chw,
            }
        );
    }
}
