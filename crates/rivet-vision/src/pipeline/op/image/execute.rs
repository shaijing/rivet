use super::{ExecutionKind, ImageOp, PipelineImageState};
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::pipeline::op::SampleContext;
use crate::pipeline::op::choice::{choice_cumulative_weights, choose_branch};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use rivet_core::DType;
use rivet_data::random::{OpKey, RandomStream};

impl ImageOp {
    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let random_key = self
            .random_key_kind()
            .map(|kind| OpKey::from_parts(kind, 0));
        self.apply_with_state(
            sample,
            ctx,
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: input_layout,
            },
            random_key,
        )
    }

    fn apply_with_state(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
        random_key: Option<OpKey>,
    ) -> RivetResult<ImageSample> {
        let input_layout = match input_state {
            PipelineImageState::Encoded => ImageAxisOrder::Hwc,
            PipelineImageState::Decoded { axis_order, .. } => axis_order,
        };
        match self {
            Self::Callback(op) => {
                let mut sample = sample.into_decoded()?;
                let mut rng = required_rng(ctx, random_key, self.name())?;
                if let Some(seed) = op.sample(&mut rng)? {
                    sample.image = op.apply("image", &sample.image, seed)?;
                }
                Ok(ImageSample::Decoded(sample))
            }
            Self::V2(op) => {
                let mut rng =
                    ctx.stream(random_key.unwrap_or_else(|| OpKey::from_parts(op.name(), 0)));
                op.apply(sample, input_layout, &mut rng)
            }
            Self::Saturation(op) => op.apply(sample),
            Self::Gamma(op) => op.apply(sample),
            Self::Affine(op) => op.apply(sample, input_layout),
            Self::Erase(op) => op.apply(sample, input_layout),
            Self::RandomRotation(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomGaussianBlur(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::AutoAugment(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::RandAugment(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::TrivialAugmentWide(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::AugMix(op) => op.apply(sample, &mut required_rng(ctx, random_key, self.name())?),
            Self::LinearTransformation(op) => op.apply(sample, input_layout),
            Self::Decode(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample, input_layout),
            Self::CenterCrop(op) => op.apply(sample, input_layout),
            Self::FiveCrop(_) | Self::TenCrop(_) => Err(invalid_pipeline(
                "multi-view crops require the expanding pipeline executor",
            )),
            Self::Pad(op) => op.apply(sample, input_layout),
            Self::Flip(op) => op.apply_with_axis_order(sample, input_layout),
            Self::RandomCrop(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomResizedCrop(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomHorizontalFlip(op) => op.apply_with_axis_order(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
            Self::Hue(op) => op.apply(sample),
            Self::ColorJitter(op) => {
                op.apply(sample, &mut required_rng(ctx, random_key, self.name())?)
            }
            Self::Invert(op) => op.apply(sample, input_layout),
            Self::Posterize(op) => op.apply(sample, input_layout),
            Self::Solarize(op) => op.apply(sample, input_layout),
            Self::Autocontrast(op) => op.apply(sample, input_layout),
            Self::Equalize(op) => op.apply(sample, input_layout),
            Self::Sharpness(op) => op.apply(sample, input_layout),
            Self::ArbitraryRotate(op) => op.apply(sample, input_layout),
            Self::RandomAffine(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::Perspective(op) => op.apply(sample, input_layout),
            Self::RandomPerspective(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::ElasticTransform(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomApply { probability, ops } => {
                let mut rng = required_rng(ctx, random_key, self.name())?;
                if !rng.gen_bool(*probability)? {
                    Ok(sample)
                } else {
                    apply_sequence(sample, ops, ctx, input_state, random_key.unwrap())
                        .map(|(sample, _)| sample)
                }
            }
            Self::RandomChoice { choices, weights } => {
                let mut rng = required_rng(ctx, random_key, self.name())?;
                let index = if choices.is_empty() {
                    return Err(invalid_argument(
                        "random_choice requires at least one choice",
                    ));
                } else {
                    let cumulative = choice_cumulative_weights(choices.len(), weights.as_deref())?;
                    choose_branch(&mut rng, choices.len(), cumulative.as_deref())?
                };
                apply_sequence(
                    sample,
                    &choices[index],
                    ctx,
                    input_state,
                    random_key
                        .unwrap()
                        .derive(OpKey::from_parts("RandomChoiceBranch", index as u32)),
                )
                .map(|(sample, _)| sample)
            }
            Self::RandomOrder { ops } => {
                let mut order: Vec<usize> = (0..ops.len()).collect();
                let mut rng = required_rng(ctx, random_key, self.name())?;
                for index in (1..order.len()).rev() {
                    let swap = rng.gen_range_usize(0..index + 1)?;
                    order.swap(index, swap);
                }
                apply_ordered_sequence(sample, ops, &order, ctx, input_state, random_key.unwrap())
                    .map(|(sample, _)| sample)
            }
            Self::GaussianBlur(op) => op.apply(sample),
            Self::Grayscale(op) => op.apply(sample, input_layout),
            Self::RandomGrayscale(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::RandomErasing(op) => op.apply(
                sample,
                &mut required_rng(ctx, random_key, self.name())?,
                input_layout,
            ),
            Self::ConvertImageDtype(_) => Err(invalid_pipeline(
                "ConvertImageDtype is a batch-stage operation",
            )),
            Self::Rotate(op) => op.apply(sample),
            Self::Normalize(op) => op.apply(sample, input_layout),
            Self::Layout(op) => op.apply(sample, input_layout),
        }
    }

    pub fn apply_sample(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
    ) -> RivetResult<ImageSample> {
        let random_key = self
            .random_key_kind()
            .map(|kind| OpKey::from_parts(kind, 0));
        self.apply_sample_with_key(sample, ctx, input_state, random_key)
    }

    pub fn apply_sample_with_key(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
        random_key: Option<OpKey>,
    ) -> RivetResult<ImageSample> {
        if self.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{} is a batch-stage operation",
                self.name()
            )));
        }
        self.apply_with_state(sample, ctx, input_state, random_key)
    }

    /// Execute a sample-stage operation from a compiled plan.
    ///
    /// Compilation has already validated the operation kind and input state,
    /// so the runtime path can dispatch directly to the transform without
    /// repeating those semantic checks.
    pub(crate) fn apply_sample_compiled(
        &self,
        sample: ImageSample,
        ctx: &SampleContext,
        input_state: PipelineImageState,
        random_key: Option<OpKey>,
    ) -> RivetResult<ImageSample> {
        self.apply_with_state(sample, ctx, input_state, random_key)
    }

    pub fn apply_batch(
        &self,
        batch: rivet_core::Tensor,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<rivet_core::Tensor> {
        match self {
            Self::V2(op) => op.apply_batch(batch, input_layout),
            Self::Normalize(op) => op.apply_batch(batch, input_layout),
            Self::ConvertImageDtype(op) => op.apply_batch(batch, input_layout),
            Self::Layout(op) => op.apply_batch(batch, input_layout),
            _ => Err(invalid_pipeline(format!(
                "{} is not a batch-stage operation",
                self.name()
            ))),
        }
    }
}

fn apply_sequence(
    mut sample: ImageSample,
    ops: &[ImageOp],
    ctx: &SampleContext,
    mut state: PipelineImageState,
    parent_key: OpKey,
) -> RivetResult<(ImageSample, PipelineImageState)> {
    for (index, op) in ops.iter().enumerate() {
        let random_key = nested_random_key(ops, index, parent_key);
        sample = op.apply_sample_with_key(sample, ctx, state, random_key)?;
        state = op.transition(state)?;
    }
    Ok((sample, state))
}

fn apply_ordered_sequence(
    mut sample: ImageSample,
    ops: &[ImageOp],
    order: &[usize],
    ctx: &SampleContext,
    mut state: PipelineImageState,
    parent_key: OpKey,
) -> RivetResult<(ImageSample, PipelineImageState)> {
    for &index in order {
        let op = &ops[index];
        let random_key = nested_random_key(ops, index, parent_key);
        sample = op.apply_sample_with_key(sample, ctx, state, random_key)?;
        state = op.transition(state)?;
    }
    Ok((sample, state))
}

fn nested_random_key(ops: &[ImageOp], index: usize, parent_key: OpKey) -> Option<OpKey> {
    let kind = ops[index].random_key_kind()?;
    let occurrence = ops[..index]
        .iter()
        .filter(|op| op.random_key_kind() == Some(kind))
        .count() as u32;
    Some(parent_key.derive(OpKey::from_parts(kind, occurrence)))
}

fn required_rng(
    ctx: &SampleContext,
    random_key: Option<OpKey>,
    op_name: &str,
) -> RivetResult<RandomStream> {
    random_key
        .map(|key| ctx.stream(key))
        .ok_or_else(|| invalid_pipeline(format!("{op_name} is missing its random key")))
}
