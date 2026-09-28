use super::builder::ImagePipeline;
use super::logical::{FusionGroupPayload, ImagePipelineContext};
use super::op::{
    BatchKernel, CompiledNormalize, CompiledProgram, CompiledSampleOp, ExecutionKind, ImageOp,
    IndexOp, PipelineImageState, SampleKernel, compile_sampler,
};
use super::physical::{BatchNode, ImageExecutionGraph, ImageGraphInfo, ImageKernel, SampleNode};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::runtime::ImageDataLoader;
use crate::sample::image::ImageAxisOrder;
use crate::sampler::IndexSampler;
use rivet_data::random::{OpKey, RandomContext};
use rivet_exec::physical::{
    ExecutableGraph, ExecutionLane, KernelStage, ParallelismClass, PhysicalGraph, PhysicalLowering,
    PhysicalNodeKind, PhysicalNodeSpec,
};
use rivet_plan::{LogicalNode, LogicalPlan, NodeId, NodeKind, OperatorStage};
use std::collections::HashMap;
use std::sync::Arc;

impl ImagePipeline {
    pub fn compile(self) -> RivetResult<ImageDataLoader> {
        self.compile_from(0)
    }
    pub fn compile_from(self, start: usize) -> RivetResult<ImageDataLoader> {
        Self::compile_logical_plan(self.to_logical_plan(), start)
    }
    /// Compile an edited logical graph, including shared inputs and joins.
    /// The graph retains the runtime/seed context from `to_logical_plan()`.
    pub fn compile_logical_plan(plan: LogicalPlan, start: usize) -> RivetResult<ImageDataLoader> {
        compile_graph(plan, start, true)
    }
    #[cfg(test)]
    pub(super) fn compile_unoptimized_for_test(self, start: usize) -> RivetResult<ImageDataLoader> {
        compile_graph(self.to_logical_plan(), start, false)
    }
}

struct VisionPhysicalLowering {
    kernels: HashMap<NodeId, (PhysicalNodeKind, Arc<ImageKernel>)>,
    placement: Option<rivet_plan::PlacementPlan>,
}
impl PhysicalLowering for VisionPhysicalLowering {
    fn lower_node(
        &self,
        id: NodeId,
        _node: &LogicalNode,
        _inputs: &[rivet_exec::physical::PhysNodeId],
    ) -> rivet_exec::runtime::RuntimeResult<PhysicalNodeSpec> {
        let (kind, operator) = self.kernels.get(&id).expect("all reachable nodes compiled");
        let lane = if *kind == PhysicalNodeKind::Source {
            ExecutionLane::Io
        } else {
            ExecutionLane::Cpu
        };
        let parallelism = match kind {
            PhysicalNodeKind::Kernel(KernelStage::Sample) => ParallelismClass::AcrossSamples,
            PhysicalNodeKind::Kernel(KernelStage::Batch) => ParallelismClass::WithinBatch,
            _ => ParallelismClass::Serial,
        };
        let mut spec = PhysicalNodeSpec::new(*kind, lane)
            .with_parallelism(parallelism)
            .with_operator(operator.clone());
        if let Some(candidate) = self.placement.as_ref().and_then(|placement| {
            placement
                .candidates
                .iter()
                .find(|candidate| candidate.node == id)
        }) {
            if candidate.device != rivet_plan::DeviceClass::Cpu {
                return Err(rivet_exec::runtime::RuntimeError::Message(
                    "vision device execution requires device-aware lowering".to_owned(),
                ));
            }
            spec.memory_requirements.output_alignment = candidate.alignment_bytes;
            spec.memory_requirements.output_contiguous =
                candidate.contiguity == rivet_plan::Contiguity::Contiguous;
            spec.memory_requirements.temporary_bytes = candidate.cost.temporary_bytes as usize;
        }
        Ok(spec)
    }
}

fn compile_graph(
    mut logical: LogicalPlan,
    start: usize,
    optimize: bool,
) -> RivetResult<ImageDataLoader> {
    logical
        .canonicalize()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let context = *logical
        .context_as::<ImagePipelineContext>()
        .ok_or_else(|| invalid_pipeline("logical plan has no vision context"))?;
    if logical
        .nodes()
        .any(|(_, node)| node.kind() == NodeKind::DeviceCut)
    {
        return Err(invalid_pipeline(
            "vision DeviceCut execution is not implemented yet",
        ));
    }
    let placement = if optimize {
        Some(
            super::optimizer::optimize_vision_plan_with_placement(
                &mut logical,
                context.runtime.num_workers,
            )?
            .1,
        )
    } else {
        None
    };
    let annotations = logical
        .infer_properties(&super::inference::VisionPropertyInference::new(
            context.runtime.num_workers,
        ))
        .map_err(super::logical::inference_error)?;
    let order = logical
        .topological_order()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let sources = order
        .iter()
        .filter_map(|id| logical.node(*id).ok()?.payload_as::<super::op::SourceOp>())
        .collect::<Vec<_>>();
    if sources.len() != 1 {
        return Err(invalid_pipeline(
            "vision graphs require one source; branches share its samples",
        ));
    }
    let source = sources[0].clone();
    let batches = order
        .iter()
        .filter_map(|id| {
            logical
                .node(*id)
                .ok()?
                .payload_as::<super::op::BatchConfig>()
        })
        .copied()
        .collect::<Vec<_>>();
    let batch = *batches
        .first()
        .ok_or_else(|| invalid_pipeline("pipeline requires .batch(size)"))?;
    batch.validate()?;
    if batches
        .iter()
        .any(|other| other.size != batch.size || other.drop_last != batch.drop_last)
    {
        return Err(invalid_pipeline(
            "DAG batch barriers must use matching size and drop_last",
        ));
    }
    let index_ops = order
        .iter()
        .filter_map(|id| logical.node(*id).ok()?.payload_as::<IndexOp>())
        .cloned()
        .collect::<Vec<_>>();
    // Index operations form one prefix immediately after the source. Branches
    // consume the same sampled indices; branch-local samplers are ambiguous.
    for id in &order {
        let node = logical
            .node(*id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        if node.kind() == NodeKind::Index {
            let input = node
                .inputs()
                .get(0)
                .ok_or_else(|| invalid_pipeline("index requires one input"))?;
            if !matches!(
                logical
                    .node(input)
                    .map_err(|error| invalid_pipeline(error.to_string()))?
                    .kind(),
                NodeKind::Source | NodeKind::Index
            ) || logical
                .children(input)
                .map_err(|error| invalid_pipeline(error.to_string()))?
                .len()
                != 1
            {
                return Err(invalid_pipeline(
                    "index operations must form a shared prefix before image branches",
                ));
            }
        }
    }
    let shuffle_seed = index_ops
        .iter()
        .find_map(|op| match op {
            IndexOp::Shuffle { seed } => Some(*seed),
            _ => None,
        })
        .unwrap_or(0);
    let random =
        RandomContext::new(context.global_seed.unwrap_or(shuffle_seed)).with_epoch(context.epoch);
    let sampler = compile_sampler(source.len(), &index_ops, random)?;
    let mut kernels = HashMap::new();
    let mut occurrences = HashMap::new();
    let mut sample_ops = Vec::new();
    let mut batch_ops = Vec::new();
    for &id in &order {
        let node = logical
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let stage = annotations
            .get(id)
            .and_then(|properties| properties.operator)
            .map(|op| op.stage)
            .unwrap_or(OperatorStage::Sample);
        let input_state = node
            .inputs()
            .get(0)
            .and_then(|input| annotations.get(input))
            .map(super::inference::state_from_properties)
            .transpose()?
            .unwrap_or(source.state());
        let (kind, kernel) = match node.kind() {
            NodeKind::Source => (
                PhysicalNodeKind::Source,
                ImageKernel::Source(source.clone()),
            ),
            NodeKind::Index => (PhysicalNodeKind::Sampler, ImageKernel::Identity),
            NodeKind::Sink => (PhysicalNodeKind::Sink, ImageKernel::Identity),
            NodeKind::Batch => (
                PhysicalNodeKind::Batch,
                ImageKernel::Batch {
                    axis_order: state_axis_order(input_state),
                },
            ),
            NodeKind::Op => {
                if let Some(concat) = node.payload_as::<super::ImageConcat>() {
                    let stage = if stage == OperatorStage::Batch {
                        KernelStage::Batch
                    } else {
                        KernelStage::Sample
                    };
                    (
                        PhysicalNodeKind::Kernel(stage),
                        ImageKernel::Concat { axis: concat.axis },
                    )
                } else if let Some(group) = node.payload_as::<FusionGroupPayload>() {
                    if group.name != "NormalizeToChw" {
                        return Err(invalid_pipeline(format!(
                            "unsupported fusion {}",
                            group.name
                        )));
                    }
                    let Some(ImageOp::Normalize(config)) = group.ops.first() else {
                        return Err(invalid_pipeline("invalid NormalizeToChw fusion"));
                    };
                    let op = BatchKernel::NormalizeToChw(CompiledNormalize {
                        config: config.clone(),
                        input_layout: ImageAxisOrder::Hwc,
                    });
                    batch_ops.push(op.clone());
                    (
                        PhysicalNodeKind::Kernel(KernelStage::Batch),
                        ImageKernel::BatchOp(Arc::new(BatchNode {
                            op,
                            output_layout: ImageAxisOrder::Chw,
                        })),
                    )
                } else {
                    let op = node
                        .payload_as::<ImageOp>()
                        .ok_or_else(|| invalid_pipeline("unknown vision op payload"))?
                        .clone();
                    op.validate()?;
                    if matches!(&op, ImageOp::Layout(config) if config.axis_order == state_axis_order(input_state))
                    {
                        (
                            PhysicalNodeKind::Kernel(KernelStage::Sample),
                            ImageKernel::Identity,
                        )
                    } else if stage == OperatorStage::Batch {
                        let input_layout = state_axis_order(input_state);
                        let kernel = match op {
                            ImageOp::Normalize(config) => {
                                BatchKernel::Normalize(CompiledNormalize {
                                    config,
                                    input_layout,
                                })
                            }
                            ImageOp::ConvertImageDtype(config) => BatchKernel::ConvertImageDtype {
                                config,
                                input_layout,
                            },
                            ImageOp::Layout(config) => BatchKernel::Layout {
                                config,
                                input_layout,
                            },
                            _ => return Err(invalid_pipeline("unsupported batch kernel")),
                        };
                        let output_layout =
                            state_axis_order(super::inference::state_from_properties(
                                annotations.get(id).expect("inferred output"),
                            )?);
                        batch_ops.push(kernel.clone());
                        (
                            PhysicalNodeKind::Kernel(KernelStage::Batch),
                            ImageKernel::BatchOp(Arc::new(BatchNode {
                                op: kernel,
                                output_layout,
                            })),
                        )
                    } else {
                        let kernel_stage = if matches!(op, ImageOp::Decode(_)) {
                            KernelStage::Decode
                        } else {
                            KernelStage::Sample
                        };
                        let compiled = if let ImageOp::Normalize(config) = op {
                            CompiledSampleOp {
                                kernel: SampleKernel::SampleNormalize(CompiledNormalize {
                                    config,
                                    input_layout: state_axis_order(input_state),
                                }),
                                random_key: None,
                                input_state,
                            }
                        } else {
                            compile_sample_op(op, input_state, None, &mut occurrences)?.0
                        };
                        sample_ops.push(compiled.clone());
                        (
                            PhysicalNodeKind::Kernel(kernel_stage),
                            ImageKernel::Sample(Arc::new(SampleNode {
                                op: compiled,
                                random,
                            })),
                        )
                    }
                }
            }
            kind => {
                return Err(invalid_pipeline(format!(
                    "unsupported vision node {kind:?}"
                )));
            }
        };
        kernels.insert(id, (kind, Arc::new(kernel)));
    }
    let mut physical = PhysicalGraph::lower(
        &logical,
        &VisionPhysicalLowering {
            kernels: kernels.clone(),
            placement,
        },
    )
    .map_err(|error| invalid_pipeline(error.to_string()))?;
    physical
        .ensure_sampler_before_source()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    // Reordering a batch barrier changes its actual input layout. Attach the
    // stack operator using that producer, rather than the logical old edge.
    let physical_batches = physical
        .nodes()
        .iter()
        .filter(|node| node.kind == PhysicalNodeKind::Batch)
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for id in physical_batches {
        let layout = physical
            .node(id)
            .ok()
            .and_then(|node| node.inputs.first())
            .and_then(|input| physical.node(*input).ok())
            .and_then(|node| node.logical_id)
            .and_then(|id| annotations.get(id))
            .map(super::inference::state_from_properties)
            .transpose()?
            .map(state_axis_order)
            .unwrap_or(state_axis_order(source.state()));
        physical
            .set_operator(id, Arc::new(ImageKernel::Batch { axis_order: layout }))
            .map_err(|error| invalid_pipeline(error.to_string()))?;
    }
    let mut sample_nodes = Vec::new();
    let mut batch_nodes = Vec::new();
    let physical_order = physical
        .execution_order()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    for id in &physical_order {
        let node = physical
            .node(*id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        if let Some((_, kernel)) = node.logical_id.and_then(|id| kernels.get(&id)) {
            match kernel.as_ref() {
                ImageKernel::Sample(node) => sample_nodes.push(Arc::clone(node)),
                ImageKernel::BatchOp(node) => batch_nodes.push(Arc::clone(node)),
                _ => {}
            }
        }
    }
    let linear = physical
        .nodes()
        .iter()
        .all(|node| node.inputs.len() <= 1 && node.last_use_count <= 1);
    let root = logical
        .root()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let output_state =
        super::inference::state_from_properties(annotations.get(root).expect("inferred root"))?;
    if logical
        .node(root)
        .map_err(|error| invalid_pipeline(error.to_string()))?
        .kind()
        != NodeKind::Sink
        || annotations
            .get(root)
            .and_then(|properties| properties.granularity)
            != Some(rivet_plan::ValueGranularity::Batch)
    {
        return Err(invalid_pipeline(
            "vision graph root must be a sink producing batches",
        ));
    }
    if output_state == PipelineImageState::Encoded {
        return Err(invalid_pipeline(
            "pipeline must decode images before batching",
        ));
    }
    let physical_batch = physical
        .nodes()
        .iter()
        .find(|node| node.kind == PhysicalNodeKind::Batch)
        .expect("batch barrier compiled");
    let pre_batch_state = physical_batch
        .inputs
        .first()
        .and_then(|input| physical.node(*input).ok())
        .and_then(|node| node.logical_id)
        .and_then(|id| annotations.get(id))
        .map(super::inference::state_from_properties)
        .transpose()?
        .unwrap_or(source.state());
    let info = ImageGraphInfo {
        input_state: source.state(),
        pre_batch_state,
        output_state,
        sample_count: sample_ops.len(),
        batch_count: batch_ops.len(),
        first_sample: sample_ops.first().map(CompiledSampleOp::name),
        first_batch: batch_ops.first().map(BatchKernel::name),
        batch_native: linear && sample_ops.is_empty() && source.supports_batch_read(),
        #[cfg(test)]
        sample_ops,
        #[cfg(test)]
        batch_ops,
    };
    let explanation = physical
        .explain()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let executable =
        ExecutableGraph::new(physical).map_err(|error| invalid_pipeline(error.to_string()))?;
    let graph = Arc::new(ImageExecutionGraph {
        executable: Arc::new(executable),
        info,
        source,
        batch,
        sample_nodes,
        batch_nodes,
        linear,
        explanation,
    });
    ImageDataLoader::new(graph, IndexSampler::new(sampler, start), context.runtime)
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
