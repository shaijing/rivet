use super::context::SampleContext;
use super::image::{ImageOp, PipelineImageState};
use super::source::SourceOp;
use crate::errors::RivetResult;
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageBatch, ImageSample};
use crate::sampler::SamplerPlan;
use rivet_data::random::{OpKey, RandomContext};

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
pub struct CompiledImageOp {
    pub op: ImageOp,
    pub random_key: Option<OpKey>,
}

impl CompiledImageOp {
    pub fn execution_kind(&self) -> super::image::ExecutionKind {
        self.op.execution_kind()
    }

    pub fn name(&self) -> &'static str {
        self.op.name()
    }
}

#[derive(Clone)]
pub struct ExecutionPlan {
    pub source: SourceOp,
    pub sampler: SamplerPlan,
    pub sample_ops: Vec<CompiledImageOp>,
    pub batch_ops: Vec<ImageOp>,
    pub batch: BatchConfig,
    /// Semantic random namespace shared by sampler and sample transforms.
    pub random: RandomContext,
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
        let ctx = SampleContext::with_random(sample_index, self.random);
        let mut state = self.input_state;
        for op in &self.sample_ops {
            sample = op
                .op
                .apply_sample_with_key(sample, &ctx, state, op.random_key)?;
            state = op.op.transition(state)?;
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
