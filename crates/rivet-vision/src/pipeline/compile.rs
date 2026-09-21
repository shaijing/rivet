use super::builder::ImagePipeline;
use super::op::{
    BatchKernel, CompiledNormalize, CompiledProgram, CompiledSampleOp, ExecutionKind,
    ExecutionPlan, ImageOp, IndexOp, PipelineImageState, SampleKernel, compile_sampler,
};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::runtime::ImageDataLoader;
use crate::sample::image::ImageAxisOrder;
use crate::sampler::IndexSampler;
use rivet_data::random::{OpKey, RandomContext};
use std::collections::HashMap;
use std::sync::Arc;

impl ImagePipeline {
    pub fn compile(self) -> RivetResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub fn compile_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        let input_state = self.source.state();
        let compiled_ops = compile_image_ops(self.ops, input_state, self.runtime.num_workers)?;

        let batch = self
            .batch
            .ok_or_else(|| invalid_pipeline("pipeline requires .batch(size)"))?;
        batch.validate()?;

        let len = self.source.len();
        let shuffle_seed = self
            .index_ops
            .iter()
            .find_map(|op| match op {
                IndexOp::Shuffle { seed } => Some(*seed),
                _ => None,
            })
            .unwrap_or(0);
        // Keep `.shuffle(seed)` as the legacy seed source when `.seed(...)`
        // was not configured, while allowing the pipeline-owned seed to be
        // independent from sampler ordering.
        let global_seed = self.global_seed.unwrap_or(shuffle_seed);
        let random = RandomContext::new(global_seed).with_epoch(self.epoch);
        let sampler = compile_sampler(len, &self.index_ops, random)?;
        let plan = ExecutionPlan {
            source: self.source,
            sampler,
            sample_ops: compiled_ops.sample_ops,
            batch_ops: compiled_ops.batch_ops,
            batch,
            random,
            input_state,
            pre_batch_state: compiled_ops.pre_batch_state,
            output_state: compiled_ops.output_state,
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

struct CompiledImageOps {
    sample_ops: Vec<CompiledSampleOp>,
    batch_ops: Vec<BatchKernel>,
    pre_batch_state: PipelineImageState,
    output_state: PipelineImageState,
}

fn compile_image_ops(
    ops: Vec<ImageOp>,
    initial_state: PipelineImageState,
    num_workers: usize,
) -> RivetResult<CompiledImageOps> {
    let mut state = initial_state;
    let mut sample_ops = Vec::new();
    let mut random_occurrences = HashMap::<&'static str, u32>::new();
    let mut batch_ops = Vec::new();
    let mut pre_batch_state = None;
    let mut batch_stage_started = false;

    let mut ops = ops.into_iter().peekable();
    while let Some(op) = ops.next() {
        op.validate()?;

        if let ImageOp::Normalize(config) = &op {
            // Normalization is expensive per pixel. When sample workers are
            // already needed for preceding augmentations, keep this work on
            // those workers and leave any following layout view for the
            // batch stage. A batch kernel is still preferable for pipelines
            // with no sample-stage work to parallelize.
            if num_workers > 0 && !sample_ops.is_empty() {
                let sample_normalize = CompiledNormalize {
                    config: config.clone(),
                    input_layout: state_axis_order(state),
                };
                let input_state = state;
                state = ImageOp::Normalize(config.clone()).transition(input_state)?;
                sample_ops.push(CompiledSampleOp {
                    kernel: SampleKernel::SampleNormalize(sample_normalize),
                    random_key: None,
                    input_state,
                });
                continue;
            }
            let can_fuse = matches!(
                state,
                PipelineImageState::Decoded {
                    dtype: rivet_core::DType::U8,
                    axis_order: crate::sample::image::ImageAxisOrder::Hwc,
                }
            ) && matches!(
                ops.peek(),
                Some(ImageOp::Layout(layout))
                    if layout.axis_order == crate::sample::image::ImageAxisOrder::Chw
            );
            if can_fuse {
                let layout = ops.next().expect("peeked fused layout operation");
                layout.validate()?;
                if !batch_stage_started {
                    pre_batch_state = Some(state);
                    batch_stage_started = true;
                }
                let fused = BatchKernel::NormalizeToChw(CompiledNormalize {
                    config: config.clone(),
                    input_layout: ImageAxisOrder::Hwc,
                });
                state = fused.transition(state)?;
                batch_ops.push(fused);
                continue;
            }
        }

        if let ImageOp::Layout(layout) = &op {
            if matches!(
                state,
                PipelineImageState::Decoded {
                    axis_order: current,
                    ..
                } if current == layout.axis_order
            ) {
                continue;
            }
        }

        match op.execution_kind() {
            ExecutionKind::Sample => {
                if batch_stage_started {
                    return Err(invalid_pipeline(format!(
                        "{} cannot follow the batch stage; move sample operations before normalize/layout (sample ops require uint8 HWC input)",
                        op.name()
                    )));
                }
                let input_state = state;
                let (compiled_op, output_state) =
                    compile_sample_op(op, input_state, None, &mut random_occurrences)?;
                state = output_state;
                sample_ops.push(compiled_op);
            }
            ExecutionKind::Batch => {
                if !batch_stage_started {
                    pre_batch_state = Some(state);
                    batch_stage_started = true;
                }
                let input_layout = state_axis_order(state);
                let kernel = match op {
                    ImageOp::Normalize(config) => BatchKernel::Normalize(CompiledNormalize {
                        config,
                        input_layout,
                    }),
                    ImageOp::ConvertImageDtype(config) => BatchKernel::ConvertImageDtype {
                        config,
                        input_layout,
                    },
                    ImageOp::Layout(config) => BatchKernel::Layout {
                        config,
                        input_layout,
                    },
                    op => unreachable!(
                        "validated batch op has sample execution kind: {}",
                        op.name()
                    ),
                };
                state = kernel.transition(state)?;
                batch_ops.push(kernel);
            }
        }
    }

    let pre_batch_state = pre_batch_state.unwrap_or(state);
    let output_state = match state {
        PipelineImageState::Encoded => Err(invalid_pipeline(
            "pipeline must decode images before batching",
        )),
        PipelineImageState::Decoded { .. } => Ok(state),
    }?;

    Ok(CompiledImageOps {
        sample_ops,
        batch_ops,
        pre_batch_state,
        output_state,
    })
}

fn compile_sample_program(
    ops: Vec<ImageOp>,
    initial_state: PipelineImageState,
    parent_key: OpKey,
) -> RivetResult<(CompiledProgram, PipelineImageState)> {
    let mut state = initial_state;
    let mut occurrences = HashMap::<&'static str, u32>::new();
    let mut compiled_ops = Vec::with_capacity(ops.len());

    for op in ops {
        let (compiled_op, output_state) =
            compile_sample_op(op, state, Some(parent_key), &mut occurrences)?;
        state = output_state;
        compiled_ops.push(compiled_op);
    }

    Ok((CompiledProgram { ops: compiled_ops }, state))
}

fn compile_sample_op(
    op: ImageOp,
    input_state: PipelineImageState,
    parent_key: Option<OpKey>,
    occurrences: &mut HashMap<&'static str, u32>,
) -> RivetResult<(CompiledSampleOp, PipelineImageState)> {
    op.validate()?;
    if op.execution_kind() != ExecutionKind::Sample {
        return Err(invalid_pipeline(format!(
            "{} cannot be compiled as a sample-stage operation",
            op.name()
        )));
    }

    let random_key = assign_random_key(&op, occurrences, parent_key);
    let (kernel, output_state, stored_random_key) = match op {
        ImageOp::RandomApply { probability, ops } => {
            let key = random_key.expect("RandomApply must have a random key");
            let (body, output_state) = compile_sample_program(ops, input_state, key)?;
            if output_state != input_state {
                return Err(invalid_pipeline(
                    "RandomApply nested transforms must preserve image state",
                ));
            }
            (
                SampleKernel::RandomApply {
                    probability,
                    key,
                    body,
                },
                input_state,
                None,
            )
        }
        ImageOp::RandomChoice { choices } => {
            let key = random_key.expect("RandomChoice must have a random key");
            if choices.is_empty() {
                return Err(invalid_pipeline(
                    "RandomChoice requires at least one choice",
                ));
            }

            let mut branches = Vec::with_capacity(choices.len());
            let mut output_state = None;
            for (index, choice) in choices.into_iter().enumerate() {
                let branch_key = key.derive(OpKey::from_parts("RandomChoiceBranch", index as u32));
                let (branch, branch_output_state) =
                    compile_sample_program(choice, input_state, branch_key)?;
                if let Some(expected_state) = output_state {
                    if expected_state != branch_output_state {
                        return Err(invalid_pipeline(
                            "random_choice choices must produce the same image state",
                        ));
                    }
                } else {
                    output_state = Some(branch_output_state);
                }
                branches.push(branch);
            }

            (
                SampleKernel::RandomChoice { key, branches },
                output_state.expect("RandomChoice choices cannot be empty"),
                None,
            )
        }
        ImageOp::RandomOrder { ops } => {
            let key = random_key.expect("RandomOrder must have a random key");
            let mut child_occurrences = HashMap::<&'static str, u32>::new();
            let mut compiled_ops = Vec::with_capacity(ops.len());
            for op in ops {
                let (compiled_op, output_state) =
                    compile_sample_op(op, input_state, Some(key), &mut child_occurrences)?;
                if output_state != input_state {
                    return Err(invalid_pipeline(
                        "RandomOrder nested transforms must preserve image state",
                    ));
                }
                compiled_ops.push(compiled_op);
            }

            (
                SampleKernel::RandomOrder {
                    key,
                    ops: compiled_ops,
                },
                input_state,
                None,
            )
        }
        op => {
            let output_state = op.transition(input_state)?;
            (SampleKernel::Semantic(op), output_state, random_key)
        }
    };

    Ok((
        CompiledSampleOp {
            kernel,
            random_key: stored_random_key,
            input_state,
        },
        output_state,
    ))
}

fn state_axis_order(state: PipelineImageState) -> ImageAxisOrder {
    match state {
        PipelineImageState::Encoded => ImageAxisOrder::Hwc,
        PipelineImageState::Decoded { axis_order, .. } => axis_order,
    }
}

fn assign_random_key(
    op: &ImageOp,
    occurrences: &mut HashMap<&'static str, u32>,
    parent_key: Option<OpKey>,
) -> Option<OpKey> {
    let kind = op.random_key_kind()?;
    let occurrence = occurrences.entry(kind).or_default();
    let local_key = OpKey::from_parts(kind, *occurrence);
    *occurrence += 1;
    Some(match parent_key {
        Some(parent_key) => parent_key.derive(local_key),
        None => local_key,
    })
}
