use super::context::SampleContext;
use super::image::{ImageOp, PipelineImageState};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::geometry::LayoutConfig;
use crate::transforms::representation::{ConvertImageDtypeConfig, NormalizeConfig};
use rivet_core::DType;
use rivet_data::random::OpKey;

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

impl CompiledNormalize {
    // Keep the planar writer's view setup out of the common per-sample
    // dispatch body, which is also used by the RGB HWC fast path.
    #[inline(never)]
    fn apply_to_chw_sample(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let decoded = sample.into_decoded()?;
        // Rank changes are views; only the final CHW result is allocated.
        let image = self
            .config
            .apply_batch_to_chw_trusted(decoded.image.unsqueeze(0)?)?
            .squeeze(0)?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: decoded.label,
        }))
    }
}

#[derive(Clone)]
pub(crate) enum SampleKernel {
    Semantic(ImageOp),
    SampleNormalize(CompiledNormalize),
    SampleNormalizeToChw(CompiledNormalize),
    RandomApply {
        probability: f64,
        key: OpKey,
        body: CompiledProgram,
    },
    RandomChoice {
        key: OpKey,
        branches: Vec<CompiledProgram>,
    },
    RandomOrder {
        key: OpKey,
        ops: Vec<CompiledSampleOp>,
    },
}

#[derive(Clone)]
pub(crate) enum BatchKernel {
    Normalize(CompiledNormalize),
    NormalizeToChw(CompiledNormalize),
    #[allow(dead_code)]
    NormalizeToChwWithCudaAugmentations {
        normalize: CompiledNormalize,
        augmentations: Vec<CompiledSampleOp>,
    },
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

#[derive(Clone)]
pub(crate) struct CompiledProgram {
    pub(crate) ops: Vec<CompiledSampleOp>,
}

impl CompiledProgram {
    pub(crate) fn execute(
        &self,
        mut sample: ImageSample,
        ctx: &SampleContext,
    ) -> RivetResult<ImageSample> {
        for op in &self.ops {
            sample = op.execute(sample, ctx)?;
        }
        Ok(sample)
    }
}

#[allow(dead_code)]
impl CompiledSampleOp {
    pub(crate) fn is_decode(&self) -> bool {
        matches!(self.kernel, SampleKernel::Semantic(ImageOp::Decode(_)))
    }

    pub(crate) fn execute(
        &self,
        mut sample: ImageSample,
        ctx: &SampleContext,
    ) -> RivetResult<ImageSample> {
        match &self.kernel {
            SampleKernel::Semantic(kernel) => {
                kernel.apply_sample_compiled(sample, ctx, self.input_state, self.random_key)
            }
            SampleKernel::SampleNormalize(kernel) => {
                kernel.config.apply_trusted(sample, kernel.input_layout)
            }
            SampleKernel::SampleNormalizeToChw(kernel) => kernel.apply_to_chw_sample(sample),
            SampleKernel::RandomApply {
                probability,
                key,
                body,
            } => {
                debug_assert!(
                    probability.is_finite() && (0.0..=1.0).contains(probability),
                    "RandomApply probability must be finite and in [0, 1]"
                );
                let mut rng = ctx.stream(*key);
                if rng.next_f64() < *probability {
                    body.execute(sample, ctx)
                } else {
                    Ok(sample)
                }
            }
            SampleKernel::RandomChoice { key, branches } => {
                debug_assert!(!branches.is_empty(), "RandomChoice must have branches");
                let mut rng = ctx.stream(*key);
                let branch = rng.gen_range_usize(0..branches.len())?;
                branches[branch].execute(sample, ctx)
            }
            SampleKernel::RandomOrder { key, ops } => {
                let mut order: Vec<usize> = (0..ops.len()).collect();
                let mut rng = ctx.stream(*key);
                for index in (1..order.len()).rev() {
                    let swap = rng.gen_range_usize(0..index + 1)?;
                    order.swap(index, swap);
                }
                for index in order {
                    sample = ops[index].execute(sample, ctx)?;
                }
                Ok(sample)
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
            SampleKernel::SampleNormalizeToChw(_) => "NormalizeToChwSample",
            SampleKernel::RandomApply { .. } => "RandomApply",
            SampleKernel::RandomChoice { .. } => "RandomChoice",
            SampleKernel::RandomOrder { .. } => "RandomOrder",
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
            Self::NormalizeToChwWithCudaAugmentations { .. } => "VisionAugmentNormalizeToChw",
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
            Self::NormalizeToChw(_) | Self::NormalizeToChwWithCudaAugmentations { .. } => {
                match input {
                    PipelineImageState::Decoded {
                        dtype: DType::U8,
                        axis_order: ImageAxisOrder::Hwc,
                    } => Ok(PipelineImageState::Decoded {
                        dtype: DType::F32,
                        axis_order: ImageAxisOrder::Chw,
                    }),
                    _ => Err(invalid_pipeline("NormalizeToChw requires U8 HWC input")),
                }
            }
        }
    }

    pub(crate) fn execute(&self, batch: rivet_core::Tensor) -> RivetResult<rivet_core::Tensor> {
        match self {
            Self::Normalize(op) => op.config.apply_batch_trusted(batch, op.input_layout),
            Self::NormalizeToChw(op) => {
                debug_assert_eq!(op.input_layout, ImageAxisOrder::Hwc);
                op.config.apply_batch_to_chw_trusted(batch)
            }
            Self::NormalizeToChwWithCudaAugmentations { .. } => Err(invalid_pipeline(
                "vision CUDA augmentation fusion must execute on its planned CUDA lane",
            )),
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
