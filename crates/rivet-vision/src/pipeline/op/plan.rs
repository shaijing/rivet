use super::context::SampleContext;
use super::image::{ImageOp, PipelineImageState};
use super::source::SourceOp;
use crate::errors::RivetResult;
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageBatch, ImageSample};
use crate::sampler::SamplerPlan;

#[derive(Clone, Copy)]
pub struct BatchConfig {
    pub size: usize,
    pub drop_last: bool,
}

impl BatchConfig {
    /// Raw configuration; pipeline compilation validates it.
    pub fn new(size: usize, drop_last: bool) -> Self {
        Self { size, drop_last }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.size == 0 {
            return Err(crate::errors::invalid_argument(
                "batch size must be greater than 0",
            ));
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct ExecutionPlan {
    pub source: SourceOp,
    pub sampler: SamplerPlan,
    pub sample_ops: Vec<ImageOp>,
    pub batch_ops: Vec<ImageOp>,
    pub batch: BatchConfig,
    /// Seed for stochastic image ops. When the pipeline shuffles, this is
    /// the shuffle seed, so one `(seed, epoch)` reproduces both the sample
    /// order and every random augmentation.
    pub random_seed: u64,
    /// Image state at the source boundary, before any operation executes.
    pub input_state: PipelineImageState,
    /// Image state after sample operations and before stacking into a batch.
    pub pre_batch_state: PipelineImageState,
    /// Compile-time image state after all compiled operations, so batch builders and bindings
    /// know the output dtype and layout before any sample is processed.
    pub output_state: PipelineImageState,
}

impl ExecutionPlan {
    pub fn can_use_batch_native(&self) -> bool {
        self.sample_ops.is_empty() && self.source.supports_batch_read()
    }

    pub fn apply_sample_ops(
        &self,
        mut sample: ImageSample,
        sample_index: usize,
    ) -> RivetResult<DecodedSample> {
        let mut ctx = SampleContext::new(sample_index);
        ctx.global_seed = self.random_seed;
        let mut state = self.input_state;
        for op in &self.sample_ops {
            let input_layout = match state {
                PipelineImageState::Decoded { axis_order, .. } => axis_order,
                PipelineImageState::Encoded => ImageAxisOrder::Hwc,
            };
            sample = op.apply_sample(sample, &mut ctx, input_layout)?;
            state = op.transition(state)?;
        }

        sample.into_decoded()
    }

    pub fn apply_batch_ops(&self, batch: ImageBatch) -> RivetResult<ImageBatch> {
        let ImageBatch { mut images, labels } = batch;
        let mut state = self.pre_batch_state;

        for op in &self.batch_ops {
            let input_layout = match state {
                PipelineImageState::Decoded { axis_order, .. } => axis_order,
                PipelineImageState::Encoded => ImageAxisOrder::Hwc,
            };
            images = op.apply_batch(images, input_layout)?;
            state = op.transition(state)?;
        }

        Ok(ImageBatch { images, labels })
    }

    pub fn stack_and_apply_batch_ops(
        &self,
        samples: impl IntoIterator<Item = DecodedSample>,
        capacity: usize,
    ) -> RivetResult<ImageBatch> {
        let mut builder = crate::batch::ImageBatchBuilder::with_capacity(capacity);
        for sample in samples {
            builder.push(sample)?;
        }
        self.apply_batch_ops(builder.finish()?)
    }
}
