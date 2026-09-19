use super::builder::ImagePipeline;
use super::op::{ExecutionPlan, ImageOp, IndexOp, PipelineImageState, compile_sampler};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::runtime::ImageDataLoader;
use crate::sampler::IndexSampler;
use std::sync::Arc;

impl ImagePipeline {
    pub fn compile(self) -> RivetResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub fn compile_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        let output_state = validate_image_ops(&self.ops, self.source.state())?;

        let batch = self
            .batch
            .ok_or_else(|| invalid_pipeline("pipeline requires .batch(size)"))?;
        batch.validate()?;

        let len = self.source.len();
        let sampler = compile_sampler(len, &self.index_ops)?;
        // Stochastic image ops share the shuffle seed when the pipeline
        // shuffles, so one seed reproduces order and augmentations; without
        // a shuffle the ops stay deterministic with a fixed seed.
        let random_seed = self
            .index_ops
            .iter()
            .find_map(|op| match op {
                IndexOp::Shuffle { seed } => Some(*seed),
                _ => None,
            })
            .unwrap_or(0);
        let plan = ExecutionPlan {
            source: self.source,
            sampler,
            ops: self.ops,
            batch,
            random_seed,
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

fn validate_image_ops(
    ops: &[ImageOp],
    initial_state: PipelineImageState,
) -> RivetResult<PipelineImageState> {
    let mut state = initial_state;

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
