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
use rivet_plan::{LogicalNode, LogicalPlan, NodeId, NodeKind, ValueGranularity};
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
        Self::compile_logical_plan_with_options(
            plan,
            start,
            super::ImageOptimizationOptions::default(),
        )
    }
    /// Compile with individually controlled logical optimization rules.
    pub fn compile_with_options(
        self,
        options: super::ImageOptimizationOptions,
    ) -> RivetResult<ImageDataLoader> {
        Self::compile_logical_plan_with_options(self.to_logical_plan(), 0, options)
    }
    pub fn compile_logical_plan_with_options(
        plan: LogicalPlan,
        start: usize,
        options: super::ImageOptimizationOptions,
    ) -> RivetResult<ImageDataLoader> {
        compile_graph(plan, start, true, options)
    }
    #[cfg(test)]
    pub(super) fn compile_unoptimized_for_test(self, start: usize) -> RivetResult<ImageDataLoader> {
        compile_graph(
            self.to_logical_plan(),
            start,
            false,
            super::ImageOptimizationOptions::default(),
        )
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
        // These typed kernels act independently on every image. Lifting
        // their contiguous suffix across stacking preserves image order and
        // padding semantics; stage classification alone grants no permission.
        if matches!(operator.as_ref(), ImageKernel::BatchOp(node) if matches!(node.op,
            BatchKernel::Normalize(_) | BatchKernel::NormalizeToChw(_) |
            BatchKernel::ConvertImageDtype { .. } | BatchKernel::Layout { .. }))
        {
            spec = spec.with_batch_lift_equivalence();
        }
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

/// Expand only an unshared terminal chain: fused sample -> Batch -> Sink.
/// This is physical implementation selection, not an IR semantic rewrite.
pub(super) fn expand_terminal_sample_fusion(
    plan: &mut LogicalPlan,
    annotations: &rivet_plan::PropertyAnnotations,
    workers: usize,
) -> RivetResult<bool> {
    if workers == 0 {
        return Ok(false);
    }
    let order = plan
        .topological_order()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    // Every reachable node is unary in a single-root graph, so the whole
    // reachable graph is one chain; no repeated consumer scans are needed.
    for &id in &order {
        if plan
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?
            .inputs()
            .len()
            > 1
        {
            return Ok(false);
        }
    }
    let Some((&sink, prefix)) = order.split_last() else {
        return Ok(false);
    };
    if plan
        .node(sink)
        .map_err(|error| invalid_pipeline(error.to_string()))?
        .kind()
        != NodeKind::Sink
    {
        return Ok(false);
    }
    let Some((&batch, prefix)) = prefix.split_last() else {
        return Ok(false);
    };
    if plan
        .node(batch)
        .map_err(|error| invalid_pipeline(error.to_string()))?
        .kind()
        != NodeKind::Batch
    {
        return Ok(false);
    }
    let Some((&group_id, preceding)) = prefix.split_last() else {
        return Ok(false);
    };
    let group_node = plan
        .node(group_id)
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let Some(group) = group_node.payload_as::<FusionGroupPayload>() else {
        return Ok(false);
    };
    let [ImageOp::Normalize(normalize), ImageOp::Layout(layout)] = group.ops.as_slice() else {
        return Ok(false);
    };
    if group.name != "NormalizeToChw" {
        return Ok(false);
    }
    let Some(input) = group_node.inputs().get(0) else {
        return Ok(false);
    };
    if !annotations.get(input).is_some_and(|properties| {
        properties.granularity == Some(ValueGranularity::Sample)
            && properties
                .operator
                .is_some_and(|operator| operator.sample_stage_has_work)
    }) {
        return Ok(false);
    }
    // Do not split across previous batch implementations or actual barriers.
    for &id in preceding {
        let node = plan
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        if matches!(node.kind(), NodeKind::Batch | NodeKind::DeviceCut)
            || node
                .payload_as::<ImageOp>()
                .is_some_and(|op| op.execution_kind() == ExecutionKind::Batch)
            || node.payload_as::<FusionGroupPayload>().is_some()
        {
            return Ok(false);
        }
    }
    let normalize = normalize.clone();
    let layout = *layout;
    let normalized = plan.add_node(LogicalNode::new(
        NodeKind::Op,
        [input],
        Some(Arc::new(ImageOp::Normalize(normalize))),
    ));
    plan.replace_node(
        group_id,
        LogicalNode::new(
            NodeKind::Op,
            [normalized],
            Some(Arc::new(ImageOp::Layout(layout))),
        ),
    )
    .map_err(|error| invalid_pipeline(error.to_string()))?;
    Ok(true)
}

fn compile_graph(
    mut logical: LogicalPlan,
    start: usize,
    optimize: bool,
    options: super::ImageOptimizationOptions,
) -> RivetResult<ImageDataLoader> {
    super::logical::assign_missing_random_identities(&mut logical)?;
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
    let (annotations, placement) = if optimize {
        let (optimizer_context, placement) = super::optimizer::optimize_vision_plan_for_execution(
            &mut logical,
            context.runtime.num_workers,
            options,
        )?;
        (optimizer_context.annotations().clone(), Some(placement))
    } else {
        (
            logical
                .infer_properties(&super::inference::VisionPropertyInference)
                .map_err(super::logical::inference_error)?,
            None,
        )
    };
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
    // Only a real, shared source prefix may execute in the source sampler.
    // A residual branch-local or post-barrier selection must never be hoisted.
    let source_id = *order
        .iter()
        .find(|id| {
            logical
                .node(**id)
                .is_ok_and(|node| node.kind() == NodeKind::Source)
        })
        .expect("one source checked");
    let mut index_ops = Vec::new();
    let mut prefix_nodes = std::collections::HashSet::new();
    let mut prefix = source_id;
    loop {
        let children = logical
            .children(prefix)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let [next] = children.as_slice() else { break };
        let node = logical
            .node(*next)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        if node.kind() != NodeKind::Index || node.inputs().len() != 1 {
            break;
        }
        let op = node
            .payload_as::<IndexOp>()
            .ok_or_else(|| invalid_pipeline("index node has incompatible payload"))?;
        index_ops.push(op.clone());
        prefix_nodes.insert(*next);
        prefix = *next;
    }
    for &id in &order {
        let node = logical
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        if node.kind() == NodeKind::Index && !prefix_nodes.contains(&id) {
            return Err(invalid_pipeline(format!(
                "index operation at node %{} remains outside the shared source prefix: branch-local or post-barrier index execution is not implemented; the optimizer retained its declared position",
                id.index()
            )));
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
    // These are implementation facts, separate from logical image properties.
    let mut physical_phases: HashMap<NodeId, (bool, bool)> = HashMap::new();
    for &id in &order {
        let node = logical
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let (mut has_sample_work, mut batch_phase) =
            node.inputs()
                .iter()
                .fold((false, false), |(work, batch), input| {
                    let (input_work, input_batch) =
                        physical_phases.get(&input).copied().unwrap_or_default();
                    (work || input_work, batch || input_batch)
                });
        let input_is_batch = node
            .inputs()
            .get(0)
            .and_then(|input| annotations.get(input))
            .is_some_and(|props| props.granularity == Some(ValueGranularity::Batch));
        let prefer_sample_normalize =
            context.runtime.num_workers > 0 && has_sample_work && !batch_phase && !input_is_batch;
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
                    let stage = if input_is_batch {
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
                    let normalize = CompiledNormalize {
                        config: config.clone(),
                        input_layout: ImageAxisOrder::Hwc,
                    };
                    if prefer_sample_normalize {
                        let compiled = CompiledSampleOp {
                            kernel: SampleKernel::SampleNormalizeToChw(normalize),
                            random_key: None,
                            input_state,
                        };
                        sample_ops.push(compiled.clone());
                        (
                            PhysicalNodeKind::Kernel(KernelStage::Sample),
                            ImageKernel::Sample(Arc::new(SampleNode {
                                op: compiled,
                                random,
                            })),
                        )
                    } else {
                        let op = BatchKernel::NormalizeToChw(normalize);
                        batch_ops.push(op.clone());
                        (
                            PhysicalNodeKind::Kernel(KernelStage::Batch),
                            ImageKernel::BatchOp(Arc::new(BatchNode {
                                op,
                                output_layout: ImageAxisOrder::Chw,
                            })),
                        )
                    }
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
                    } else if op.execution_kind() == ExecutionKind::Batch
                        && !(matches!(op, ImageOp::Normalize(_)) && prefer_sample_normalize)
                    {
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
                            compile_sample_op(
                                op,
                                input_state,
                                None,
                                &mut occurrences,
                                node.semantic_identity().map(OpKey::from_u64),
                            )?
                            .0
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
        match kind {
            PhysicalNodeKind::Batch | PhysicalNodeKind::Kernel(KernelStage::Batch) => {
                batch_phase = true
            }
            PhysicalNodeKind::Kernel(KernelStage::Decode | KernelStage::Sample) => {
                has_sample_work |= matches!(kernel, ImageKernel::Sample(_));
            }
            _ => {}
        }
        physical_phases.insert(id, (has_sample_work, batch_phase));
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
    let linear_topology = physical
        .nodes()
        .iter()
        .all(|node| node.inputs.len() <= 1 && node.last_use_count <= 1);
    // The typed pull executor partitions sample and batch kernels. Use that
    // traversal only if the physical graph has already proved this ordering;
    // an unlifted batch kernel must execute in place through the graph executor.
    let mut saw_batch = false;
    let linear = linear_topology
        && physical_order.iter().all(|id| {
            let node = physical.node(*id).expect("validated execution order");
            match node.kind {
                PhysicalNodeKind::Batch => {
                    if saw_batch {
                        return false;
                    }
                    saw_batch = true;
                    true
                }
                PhysicalNodeKind::Kernel(KernelStage::Decode | KernelStage::Sample) => !saw_batch,
                PhysicalNodeKind::Kernel(KernelStage::Batch) => saw_batch,
                _ => true,
            }
        });
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
            compile_sample_op(op, state, Some(parent_key), &mut occurrences, None)?;
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
    stable_key: Option<OpKey>,
) -> RivetResult<(CompiledSampleOp, PipelineImageState)> {
    op.validate()?;
    if op.execution_kind() != ExecutionKind::Sample {
        return Err(invalid_pipeline(format!(
            "{} cannot be compiled as a sample-stage operation",
            op.name()
        )));
    }

    let random_key = stable_key.or_else(|| assign_random_key(&op, occurrences, parent_key));
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
                    compile_sample_op(op, input_state, Some(key), &mut child_occurrences, None)?;
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

#[cfg(test)]
mod implementation_tests {
    use super::*;
    use crate::cache::DenseImageMemoryDataset;
    use crate::source::ImageSource;
    use rivet_core::{DType, Device, Tensor};

    fn pipeline(workers: usize) -> ImagePipeline {
        let source = DenseImageMemoryDataset::new(
            Tensor::from_vec(
                (0..96).map(|i| i as u8).collect::<Vec<_>>(),
                [2, 4, 4, 3],
                &Device::Cpu,
            )
            .unwrap(),
            Tensor::from_vec(vec![3i64, 7], [2], &Device::Cpu).unwrap(),
        )
        .unwrap();
        ImagePipeline::from_source(ImageSource::from_dense_decoded(Arc::new(source)))
            .convert_image_dtype(DType::U8)
            .crop(1, 1, 2, 2)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .workers(workers)
            .batch(2, false)
    }

    #[test]
    fn execution_preparation_places_once_and_keeps_semantic_explain_separate() {
        let mut declared = pipeline(2).to_logical_plan();
        let (context, placement) = super::super::optimizer::optimize_vision_plan_for_execution(
            &mut declared,
            2,
            super::super::ImageOptimizationOptions::default(),
        )
        .unwrap();
        assert!(
            !context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "placement.explain")
        );
        assert!(
            !declared
                .nodes()
                .any(|(_, node)| node.payload_as::<FusionGroupPayload>().is_some())
        );
        assert_eq!(
            placement.candidates.len(),
            declared.topological_order().unwrap().len()
        );
        let mut semantic = pipeline(2).to_logical_plan();
        let (context, _) = super::super::optimizer::optimize_vision_plan_with_options(
            &mut semantic,
            2,
            super::super::ImageOptimizationOptions::default(),
        )
        .unwrap();
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "placement.explain")
        );
        assert!(
            semantic
                .nodes()
                .any(|(_, node)| node.payload_as::<FusionGroupPayload>().is_some())
        );
    }

    #[test]
    fn identity_conversion_crop_and_fused_normalize_preserve_values() {
        for workers in [0, 2] {
            let mut optimized = pipeline(workers).compile().unwrap();
            if workers > 0 {
                assert_eq!(
                    optimized.info.sample_ops.last().unwrap().name(),
                    "NormalizeSample"
                );
                assert_eq!(optimized.info.first_batch_op_name(), Some("Layout"));
                assert_eq!(
                    optimized.info.pre_batch_state,
                    PipelineImageState::Decoded {
                        dtype: DType::F32,
                        axis_order: ImageAxisOrder::Hwc,
                    }
                );
            } else {
                assert_eq!(optimized.info.first_batch_op_name(), Some("NormalizeToChw"));
            }
            let mut original = pipeline(workers).compile_unoptimized_for_test(0).unwrap();
            let optimized = optimized.next_batch().unwrap().unwrap();
            let original = original.next_batch().unwrap().unwrap();
            assert_eq!(optimized.images.dims(), [2, 3, 2, 2]);
            assert_eq!(
                optimized.images.to_vec::<f32>().unwrap(),
                original.images.to_vec::<f32>().unwrap()
            );
            assert_eq!(
                optimized.labels.to_vec::<i64>().unwrap(),
                original.labels.to_vec::<i64>().unwrap()
            );
        }
    }
}
