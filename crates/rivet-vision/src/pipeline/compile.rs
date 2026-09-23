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
use rivet_exec::physical::{
    ExecutionLane, KernelStage, PhysicalGraph, PhysicalLowering, PhysicalNodeKind, PhysicalNodeSpec,
};
use rivet_plan::{LogicalNode, LogicalPlan, NodeId, NodeKind, OperatorStage, PropertyAnnotations};
use std::collections::HashMap;
use std::sync::Arc;

impl ImagePipeline {
    pub fn compile(self) -> RivetResult<ImageDataLoader> {
        self.compile_from(0)
    }

    pub fn compile_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        #[cfg(not(feature = "cuda"))]
        if self.runtime.sink_device_ordinal.is_some() {
            return Err(invalid_pipeline(
                "CUDA sink requested, but rivet-vision was built without the cuda feature",
            ));
        }
        let enable_cuda_augmentation_fusion =
            self.runtime.sink_device_ordinal.is_some() && self.source.supports_batch_read();
        let mut logical = self.to_logical_plan();
        let (_, placement) = super::optimizer::optimize_vision_plan_for_sink(
            &mut logical,
            self.runtime.num_workers,
            self.runtime.sink_device_ordinal,
        )?;
        let physical = lower_vision_physical(
            &logical,
            self.runtime.num_workers,
            &placement,
            self.runtime.sink_device_ordinal,
        )?;
        Self::from_logical_plan(&logical)?.compile_legacy_from_physical(
            start,
            physical,
            enable_cuda_augmentation_fusion,
        )
    }

    #[cfg(test)]
    fn compile_legacy_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        let enable_cuda_augmentation_fusion =
            self.runtime.sink_device_ordinal.is_some() && self.source.supports_batch_read();
        let mut logical = self.to_logical_plan();
        let (_, placement) = super::optimizer::optimize_vision_plan_for_sink(
            &mut logical,
            self.runtime.num_workers,
            self.runtime.sink_device_ordinal,
        )?;
        let physical = lower_vision_physical(
            &logical,
            self.runtime.num_workers,
            &placement,
            self.runtime.sink_device_ordinal,
        )?;
        self.compile_legacy_from_physical(start, physical, enable_cuda_augmentation_fusion)
    }

    fn compile_legacy_from_physical(
        self,
        start: usize,
        physical: PhysicalGraph,
        enable_cuda_augmentation_fusion: bool,
    ) -> RivetResult<ImageDataLoader> {
        let sink_device = match self.runtime.sink_device_ordinal {
            None => None,
            Some(ordinal) => {
                #[cfg(feature = "cuda")]
                {
                    Some(rivet_core::Device::cuda(ordinal)?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    return Err(invalid_pipeline(
                        "CUDA sink requested, but rivet-vision was built without the cuda feature",
                    ));
                }
            }
        };
        let input_state = self.source.state();
        let cuda_batch_kernel = physical.nodes().iter().any(|node| {
            node.kind == PhysicalNodeKind::Kernel(KernelStage::Batch)
                && matches!(node.lane, ExecutionLane::Device { .. })
        });
        let compiled_ops = compile_image_ops(
            self.ops,
            input_state,
            self.runtime.num_workers,
            cuda_batch_kernel && enable_cuda_augmentation_fusion,
        )?;

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
            physical,
            sink_device,
        )
    }

    #[cfg(test)]
    pub(super) fn compile_legacy_for_test(self, start: usize) -> RivetResult<ImageDataLoader> {
        self.compile_legacy_from(start)
    }
}

struct VisionPhysicalLowering {
    annotations: PropertyAnnotations,
    placement: rivet_plan::PlacementPlan,
    sink_device_ordinal: Option<usize>,
}

impl PhysicalLowering for VisionPhysicalLowering {
    fn lower_node(
        &self,
        logical_id: NodeId,
        node: &LogicalNode,
        _physical_inputs: &[rivet_exec::physical::PhysNodeId],
    ) -> rivet_exec::runtime::RuntimeResult<PhysicalNodeSpec> {
        let (kind, default_lane) = match node.kind() {
            NodeKind::Source => (PhysicalNodeKind::Source, ExecutionLane::Io),
            NodeKind::Index => (PhysicalNodeKind::Sampler, ExecutionLane::Cpu),
            NodeKind::Op => {
                let stage = self
                    .annotations
                    .get(logical_id)
                    .and_then(|properties| properties.operator)
                    .map(|operator| operator.stage)
                    .unwrap_or(OperatorStage::Sample);
                let kernel = match stage {
                    OperatorStage::Batch => KernelStage::Batch,
                    OperatorStage::Sample | OperatorStage::Source => KernelStage::Sample,
                };
                (PhysicalNodeKind::Kernel(kernel), ExecutionLane::Cpu)
            }
            NodeKind::Batch => (PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            NodeKind::Cache => (PhysicalNodeKind::Cache, ExecutionLane::Io),
            NodeKind::Sink => (PhysicalNodeKind::Sink, ExecutionLane::Cpu),
        };
        let selected_device = self
            .placement
            .candidates
            .iter()
            .find(|candidate| candidate.node == logical_id)
            .map(|candidate| &candidate.device);
        let lane = match selected_device {
            Some(rivet_plan::DeviceClass::Cuda) if node.kind() == NodeKind::Sink => {
                ExecutionLane::Device {
                    ordinal: self.sink_device_ordinal.ok_or_else(|| {
                        rivet_exec::runtime::RuntimeError::Message(
                            "CUDA sink placement has no requested device ordinal".to_owned(),
                        )
                    })?,
                }
            }
            Some(rivet_plan::DeviceClass::Cuda)
                if kind == PhysicalNodeKind::Kernel(KernelStage::Batch) =>
            {
                ExecutionLane::Device {
                    ordinal: self.sink_device_ordinal.ok_or_else(|| {
                        rivet_exec::runtime::RuntimeError::Message(
                            "CUDA batch kernel placement has no requested device ordinal"
                                .to_owned(),
                        )
                    })?,
                }
            }
            Some(rivet_plan::DeviceClass::Cuda) => {
                return Err(rivet_exec::runtime::RuntimeError::Message(format!(
                    "CUDA execution for logical node %{} is not implemented in this phase",
                    logical_id.index()
                )));
            }
            _ => default_lane,
        };
        Ok(PhysicalNodeSpec::new(kind, lane))
    }
}

fn lower_vision_physical(
    logical: &LogicalPlan,
    workers: usize,
    placement: &rivet_plan::PlacementPlan,
    sink_device_ordinal: Option<usize>,
) -> RivetResult<PhysicalGraph> {
    let annotations = logical
        .infer_properties(&super::inference::VisionPropertyInference::new(workers))
        .map_err(super::logical::inference_error)?;
    let mut physical = PhysicalGraph::lower(
        logical,
        &VisionPhysicalLowering {
            annotations,
            placement: placement.clone(),
            sink_device_ordinal,
        },
    )
    .map_err(|error| invalid_pipeline(format!("physical lowering failed: {error}")))?;
    physical
        .ensure_sampler_before_source()
        .map_err(|error| invalid_pipeline(format!("physical sampler planning failed: {error}")))?;
    let transfers = physical
        .insert_transfers_for_lane_changes()
        .map_err(|error| invalid_pipeline(format!("physical transfer planning failed: {error}")))?;
    for transfer in transfers {
        let consumer_logical_id = physical
            .nodes()
            .iter()
            .find(|node| node.inputs.contains(&transfer))
            .and_then(|node| node.logical_id);
        if let Some(estimate) = consumer_logical_id
            .and_then(|logical_id| {
                placement
                    .candidates
                    .iter()
                    .find(|candidate| candidate.node == logical_id)
            })
            .map(|candidate| candidate.cost.transfer_bytes)
        {
            physical
                .set_transfer_estimate(transfer, estimate)
                .map_err(|error| {
                    invalid_pipeline(format!("physical transfer costing failed: {error}"))
                })?;
        }
    }
    Ok(physical)
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
    defer_cuda_augmentations: bool,
) -> RivetResult<CompiledImageOps> {
    let mut state = initial_state;
    let mut sample_ops = Vec::new();
    let mut random_occurrences = HashMap::<&'static str, u32>::new();
    let mut batch_ops = Vec::new();
    let mut cuda_augmentations = Vec::new();
    let mut pre_batch_state = None;
    let mut batch_stage_started = false;

    let deferred_range = if defer_cuda_augmentations {
        cuda_augmentation_prefix(&ops)
    } else {
        None
    };

    let mut ops = ops.into_iter().enumerate().peekable();
    while let Some((op_index, op)) = ops.next() {
        op.validate()?;

        if deferred_range
            .as_ref()
            .is_some_and(|range| range.contains(&op_index))
        {
            let input_state = state;
            let (compiled_op, output_state) =
                compile_sample_op(op, input_state, None, &mut random_occurrences)?;
            state = output_state;
            cuda_augmentations.push(compiled_op);
            continue;
        }

        if let ImageOp::Normalize(config) = &op {
            // Normalization is expensive per pixel. When sample workers are
            // already needed for preceding augmentations, keep this work on
            // those workers and leave any following layout view for the
            // batch stage. A batch kernel is still preferable for pipelines
            // with no sample-stage work to parallelize.
            if num_workers > 0 && !sample_ops.is_empty() && cuda_augmentations.is_empty() {
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
                ops.peek().map(|(_, op)| op),
                Some(ImageOp::Layout(layout))
                    if layout.axis_order == crate::sample::image::ImageAxisOrder::Chw
            );
            if can_fuse {
                let (_, layout) = ops.next().expect("peeked fused layout operation");
                layout.validate()?;
                if !batch_stage_started {
                    pre_batch_state = Some(state);
                    batch_stage_started = true;
                }
                let normalize = CompiledNormalize {
                    config: config.clone(),
                    input_layout: ImageAxisOrder::Hwc,
                };
                let fused = if cuda_augmentations.is_empty() {
                    BatchKernel::NormalizeToChw(normalize)
                } else {
                    BatchKernel::NormalizeToChwWithCudaAugmentations {
                        normalize,
                        augmentations: cuda_augmentations.clone(),
                    }
                };
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

fn cuda_augmentation_prefix(ops: &[ImageOp]) -> Option<std::ops::Range<usize>> {
    let normalize_index = ops.len().checked_sub(2)?;
    if !matches!(ops.get(normalize_index), Some(ImageOp::Normalize(_)))
        || !matches!(ops.get(normalize_index + 1), Some(ImageOp::Layout(layout)) if layout.axis_order == ImageAxisOrder::Chw)
    {
        return None;
    }
    let mut start = normalize_index;
    while start > 0 && ops.get(start - 1).is_some_and(is_cuda_augmentation) {
        start -= 1;
    }
    if start == normalize_index {
        return None;
    }
    if ops[start..normalize_index]
        .iter()
        .filter(|op| matches!(op, ImageOp::RandomResizedCrop(_)))
        .count()
        > 1
    {
        return None;
    }
    Some(start..normalize_index)
}

fn is_cuda_augmentation(op: &ImageOp) -> bool {
    matches!(
        op,
        ImageOp::Flip(_) | ImageOp::RandomHorizontalFlip(_) | ImageOp::RandomResizedCrop(_)
    )
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
