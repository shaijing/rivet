use super::builder::ImagePipeline;
use super::op::{
    CompiledImageOp, ExecutionKind, ExecutionPlan, ImageOp, IndexOp, PipelineImageState,
    compile_sampler,
};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::runtime::ImageDataLoader;
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
        let compiled_ops = compile_image_ops(self.ops, input_state)?;

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
    sample_ops: Vec<CompiledImageOp>,
    batch_ops: Vec<ImageOp>,
    pre_batch_state: PipelineImageState,
    output_state: PipelineImageState,
}

fn compile_image_ops(
    ops: Vec<ImageOp>,
    initial_state: PipelineImageState,
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
                let fused = ImageOp::NormalizeToChw(config.clone());
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
                state = op.transition(state)?;
                let random_key = assign_random_key(&op, &mut random_occurrences);
                sample_ops.push(CompiledImageOp { op, random_key });
            }
            ExecutionKind::Batch => {
                if !batch_stage_started {
                    pre_batch_state = Some(state);
                    batch_stage_started = true;
                }
                state = op.transition(state)?;
                batch_ops.push(op);
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

fn assign_random_key(op: &ImageOp, occurrences: &mut HashMap<&'static str, u32>) -> Option<OpKey> {
    let kind = op.random_key_kind()?;
    let occurrence = occurrences.entry(kind).or_default();
    let key = OpKey::from_parts(kind, *occurrence);
    *occurrence += 1;
    Some(key)
}
