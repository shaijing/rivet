use super::context::SampleContext;
use super::image::{ImageOp, PipelineImageState};
use super::source::SourceOp;
use crate::errors::RivetResult;
use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
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
    pub ops: Vec<ImageOp>,
    pub batch: BatchConfig,
    /// Seed for stochastic image ops. When the pipeline shuffles, this is
    /// the shuffle seed, so one `(seed, epoch)` reproduces both the sample
    /// order and every random augmentation.
    pub random_seed: u64,
    /// Compile-time image state after `ops`, so batch builders and bindings
    /// know the output dtype and layout before any sample is processed.
    pub output_state: PipelineImageState,
}

impl ExecutionPlan {
    pub fn apply_ops(
        &self,
        mut sample: ImageSample,
        sample_index: usize,
    ) -> RivetResult<DecodedSample> {
        let mut ctx = SampleContext::new(sample_index);
        ctx.global_seed = self.random_seed;
        let mut state = self.source.state();
        for op in &self.ops {
            let input_layout = match state {
                PipelineImageState::Decoded { layout, .. } => layout,
                PipelineImageState::Encoded => ImageLayout::Hwc,
            };
            sample = op.apply(sample, &mut ctx, input_layout)?;
            state = op.transition(state)?;
        }

        sample.into_decoded()
    }
}
