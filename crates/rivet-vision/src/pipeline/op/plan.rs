use super::context::SampleContext;
use super::image::{ImageOp, PipelineImageState};
use super::source::SourceOp;
use crate::errors::{invalid_pipeline, RivetResult};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageBatch, ImageSample};
use crate::sampler::SamplerPlan;
use crate::transforms::geometry::LayoutConfig;
use crate::transforms::representation::{ConvertImageDtypeConfig, NormalizeConfig};
use rivet_core::DType;
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
pub(crate) struct CompiledNormalize {
    pub(crate) config: NormalizeConfig,
    pub(crate) input_layout: ImageAxisOrder,
}

#[derive(Clone)]
pub(crate) enum SampleKernel {
    Semantic(ImageOp),
    SampleNormalize(CompiledNormalize),
}

#[derive(Clone)]
pub(crate) enum BatchKernel {
    Normalize(CompiledNormalize),
    NormalizeToChw(CompiledNormalize),
    ConvertImageDtype {
        config: ConvertImageDtypeConfig,
        input_layout: ImageAxisOrder,
    },
    Layout {
        config: LayoutConfig,
        input_layout: ImageAxisOrder,
    },
}

#[derive(Clone)]
pub(crate) struct CompiledSampleOp {
    pub(crate) kernel: SampleKernel,
    pub(crate) random_key: Option<OpKey>,
    pub(crate) input_state: PipelineImageState,
}

#[allow(dead_code)]
impl CompiledSampleOp {
    pub(crate) fn execute(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
    ) -> RivetResult<ImageSample> {
        match &self.kernel {
            SampleKernel::Semantic(kernel) => {
                kernel.apply_sample_compiled(sample, ctx, self.input_state, self.random_key)
            }
            SampleKernel::SampleNormalize(kernel) => {
                kernel.config.apply_trusted(sample, kernel.input_layout)
            }
        }
    }

    pub(crate) fn execution_kind(&self) -> super::image::ExecutionKind {
        super::image::ExecutionKind::Sample
    }

    pub(crate) fn name(&self) -> &'static str {
        match &self.kernel {
            SampleKernel::Semantic(op) => op.name(),
            SampleKernel::SampleNormalize(_) => "NormalizeSample",
        }
    }
}

#[allow(dead_code)]
impl BatchKernel {
    pub(crate) fn execution_kind(&self) -> super::image::ExecutionKind {
        super::image::ExecutionKind::Batch
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Normalize(_) => "Normalize",
            Self::NormalizeToChw(_) => "NormalizeToChw",
            Self::ConvertImageDtype { .. } => "ConvertImageDtype",
            Self::Layout { .. } => "Layout",
        }
    }

    pub(crate) fn transition(&self, input: PipelineImageState) -> RivetResult<PipelineImageState> {
        match self {
            Self::Normalize(op) => ImageOp::Normalize(op.config.clone()).transition(input),
            Self::ConvertImageDtype { config, .. } => {
                ImageOp::ConvertImageDtype(config.clone()).transition(input)
            }
            Self::Layout { config, .. } => ImageOp::Layout(config.clone()).transition(input),
            Self::NormalizeToChw(_) => match input {
                PipelineImageState::Decoded {
                    dtype: DType::U8,
                    axis_order: ImageAxisOrder::Hwc,
                } => Ok(PipelineImageState::Decoded {
                    dtype: DType::F32,
                    axis_order: ImageAxisOrder::Chw,
                }),
                _ => Err(invalid_pipeline("NormalizeToChw requires U8 HWC input")),
            },
        }
    }

    pub(crate) fn execute(&self, batch: rivet_core::Tensor) -> RivetResult<rivet_core::Tensor> {
        match self {
            Self::Normalize(op) => op.config.apply_batch_trusted(batch, op.input_layout),
            Self::NormalizeToChw(op) => {
                debug_assert_eq!(op.input_layout, ImageAxisOrder::Hwc);
                op.config.apply_batch_to_chw_trusted(batch)
            }
            Self::ConvertImageDtype {
                config,
                input_layout,
            } => config.apply_batch_trusted(batch, *input_layout),
            Self::Layout {
                config,
                input_layout,
            } => config.apply_batch_trusted(batch, *input_layout),
        }
    }
}

#[derive(Clone)]
pub struct ExecutionPlan {
    pub source: SourceOp,
    pub sampler: SamplerPlan,
    pub(crate) sample_ops: Vec<CompiledSampleOp>,
    pub(crate) batch_ops: Vec<BatchKernel>,
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
    /// Number of sample kernels produced by the compiler.
    pub fn sample_op_count(&self) -> usize {
        self.sample_ops.len()
    }

    /// Number of batch kernels produced by the compiler.
    pub fn batch_op_count(&self) -> usize {
        self.batch_ops.len()
    }

    /// Name of the first compiled sample kernel, for diagnostics.
    pub fn first_sample_op_name(&self) -> Option<&'static str> {
        self.sample_ops.first().map(CompiledSampleOp::name)
    }

    /// Name of the first compiled batch kernel, for diagnostics.
    pub fn first_batch_op_name(&self) -> Option<&'static str> {
        self.batch_ops.first().map(BatchKernel::name)
    }

    pub fn can_use_batch_native(&self) -> bool {
        self.sample_ops.is_empty() && self.source.supports_batch_read()
    }

    pub fn apply_sample_ops(
        &self,
        mut sample: ImageSample,
        sample_index: usize,
    ) -> RivetResult<DecodedSample> {
        let ctx = SampleContext::with_random(sample_index, self.random);
        for op in &self.sample_ops {
            sample = op.execute(sample, &ctx)?;
        }

        sample.into_decoded()
    }

    pub fn apply_batch_ops(&self, batch: ImageBatch) -> RivetResult<ImageBatch> {
        let ImageBatch {
            mut images,
            labels,
            axis_order: batch_axis_order,
        } = batch;

        for op in &self.batch_ops {
            images = op.execute(images)?;
        }

        let output_axis_order = state_axis_order(self.output_state);
        debug_assert_eq!(batch_axis_order, state_axis_order(self.pre_batch_state));

        Ok(ImageBatch {
            images,
            labels,
            axis_order: output_axis_order,
        })
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

    pub fn stack_and_apply_batch_results(
        &self,
        samples: impl IntoIterator<Item = RivetResult<DecodedSample>>,
        capacity: usize,
    ) -> RivetResult<ImageBatch> {
        let mut builder = crate::batch::ImageBatchBuilder::with_capacity(capacity);
        for sample in samples {
            builder.push(sample?)?;
        }
        self.apply_batch_ops(builder.finish()?)
    }
}

fn state_axis_order(state: PipelineImageState) -> ImageAxisOrder {
    match state {
        PipelineImageState::Encoded => ImageAxisOrder::Hwc,
        PipelineImageState::Decoded { axis_order, .. } => axis_order,
    }
}
