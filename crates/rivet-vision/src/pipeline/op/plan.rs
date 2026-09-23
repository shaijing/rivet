use super::context::SampleContext;
use super::image::{ImageOp, PipelineImageState};
use super::source::SourceOp;
use crate::errors::{RivetResult, invalid_pipeline};
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

    #[cfg(feature = "cuda")]
    pub(crate) fn apply_cuda_batch_ops(
        &self,
        batch: ImageBatch,
        indices: &[usize],
    ) -> RivetResult<ImageBatch> {
        let (op, augmentations) = match self.batch_ops.as_slice() {
            [BatchKernel::NormalizeToChw(op)] => (op, &[][..]),
            [
                BatchKernel::NormalizeToChwWithCudaAugmentations {
                    normalize,
                    augmentations,
                },
            ] => (normalize, augmentations.as_slice()),
            _ => {
                return Err(invalid_pipeline(
                    "CUDA batch execution requires one fused NormalizeToChw kernel",
                ));
            }
        };
        if batch.axis_order != ImageAxisOrder::Hwc {
            return Err(invalid_pipeline(
                "CUDA NormalizeToChw kernel requires an NHWC input batch",
            ));
        }
        let images = if augmentations.is_empty() {
            crate::transforms::representation::normalize_u8_batch_to_nchw_f32_cuda(
                &batch.images,
                &op.config.mean,
                &op.config.std,
            )?
        } else {
            let (output_height, output_width, params, filter) =
                cuda_augmentation_params(self, augmentations, &batch.images, indices)?;
            let (scale, bias) = op.config.expanded_affine(3)?;
            batch.images.cuda_augment_normalize_u8_nhwc_to_nchw_f32(
                output_height,
                output_width,
                &params,
                filter,
                &scale,
                &bias,
            )?
        };
        Ok(ImageBatch {
            images,
            labels: batch.labels,
            axis_order: ImageAxisOrder::Chw,
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

#[cfg(feature = "cuda")]
fn cuda_augmentation_params(
    plan: &ExecutionPlan,
    augmentations: &[CompiledSampleOp],
    input: &rivet_core::Tensor,
    indices: &[usize],
) -> RivetResult<(usize, usize, Vec<u32>, i32)> {
    use crate::transforms::CropRegion;
    use crate::transforms::geometry::FlipDirection;

    let dims = input.dims();
    if dims.len() != 4 || dims[3] != 3 || dims[0] != indices.len() {
        return Err(invalid_pipeline(format!(
            "CUDA image augmentations require a matching NHWC RGB batch and source indices, got shape {dims:?} and {} indices",
            indices.len()
        )));
    }
    let (input_height, input_width) = (dims[1], dims[2]);
    let (mut output_height, mut output_width) = (input_height, input_width);
    let mut params = Vec::with_capacity(indices.len().saturating_mul(8));
    let mut filter = 1;
    for &sample_index in indices {
        let mut crop = CropRegion {
            x: 0,
            y: 0,
            width: u32::try_from(input_width)
                .map_err(|_| invalid_pipeline("CUDA image width exceeds u32"))?,
            height: u32::try_from(input_height)
                .map_err(|_| invalid_pipeline("CUDA image height exceeds u32"))?,
        };
        let mut source_reverse_x = false;
        let mut source_reverse_y = false;
        let mut output_flip_x = false;
        let mut output_flip_y = false;
        let mut has_crop = false;

        for op in augmentations {
            let SampleKernel::Semantic(image_op) = &op.kernel else {
                return Err(invalid_pipeline(format!(
                    "{} cannot be lowered into the CUDA vision augmentation kernel",
                    op.name()
                )));
            };
            match image_op {
                ImageOp::Flip(config) => {
                    let orientation = if has_crop {
                        (&mut output_flip_x, &mut output_flip_y)
                    } else {
                        (&mut source_reverse_x, &mut source_reverse_y)
                    };
                    match config.direction {
                        FlipDirection::Horizontal => *orientation.0 = !*orientation.0,
                        FlipDirection::Vertical => *orientation.1 = !*orientation.1,
                    }
                }
                ImageOp::RandomHorizontalFlip(config) => {
                    let key = op.random_key.ok_or_else(|| {
                        invalid_pipeline("RandomHorizontalFlip has no compiled OpKey")
                    })?;
                    let ctx = SampleContext::with_random(sample_index, plan.random);
                    let mut rng = ctx.stream(key);
                    if rng.gen_bool(config.probability)? {
                        if has_crop {
                            output_flip_x = !output_flip_x;
                        } else {
                            source_reverse_x = !source_reverse_x;
                        }
                    }
                }
                ImageOp::RandomResizedCrop(config) => {
                    if has_crop {
                        return Err(invalid_pipeline(
                            "CUDA fusion currently supports one RandomResizedCrop per fused batch",
                        ));
                    }
                    let key = op.random_key.ok_or_else(|| {
                        invalid_pipeline("RandomResizedCrop has no compiled OpKey")
                    })?;
                    let ctx = SampleContext::with_random(sample_index, plan.random);
                    let mut rng = ctx.stream(key);
                    let region = config.resolve(crop.width.max(1), crop.height.max(1), &mut rng);
                    let source_width = u32::try_from(input_width)
                        .map_err(|_| invalid_pipeline("CUDA image width exceeds u32"))?;
                    let source_height = u32::try_from(input_height)
                        .map_err(|_| invalid_pipeline("CUDA image height exceeds u32"))?;
                    crop = CropRegion {
                        x: if source_reverse_x {
                            source_width - region.x - region.width
                        } else {
                            region.x
                        },
                        y: if source_reverse_y {
                            source_height - region.y - region.height
                        } else {
                            region.y
                        },
                        width: region.width,
                        height: region.height,
                    };
                    output_flip_x = false;
                    output_flip_y = false;
                    output_width = config.width as usize;
                    output_height = config.height as usize;
                    filter = interpolation_id(config.interpolation);
                    has_crop = true;
                }
                _ => {
                    return Err(invalid_pipeline(format!(
                        "{} is not supported by the CUDA vision augmentation fusion",
                        image_op.name()
                    )));
                }
            }
        }

        if !has_crop {
            output_flip_x = source_reverse_x;
            output_flip_y = source_reverse_y;
            source_reverse_x = false;
            source_reverse_y = false;
        }
        params.extend_from_slice(&[
            crop.x,
            crop.y,
            crop.width,
            crop.height,
            u32::from(source_reverse_x),
            u32::from(source_reverse_y),
            u32::from(output_flip_x),
            u32::from(output_flip_y),
        ]);
    }
    Ok((output_height, output_width, params, filter))
}

#[cfg(feature = "cuda")]
fn interpolation_id(interpolation: crate::transforms::InterpolationMode) -> i32 {
    match interpolation {
        crate::transforms::InterpolationMode::Nearest => 0,
        crate::transforms::InterpolationMode::Bilinear => 1,
        crate::transforms::InterpolationMode::Bicubic => 2,
        crate::transforms::InterpolationMode::Lanczos3 => 3,
    }
}

fn state_axis_order(state: PipelineImageState) -> ImageAxisOrder {
    match state {
        PipelineImageState::Encoded => ImageAxisOrder::Hwc,
        PipelineImageState::Decoded { axis_order, .. } => axis_order,
    }
}
