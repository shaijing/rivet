use super::physical::{BatchNode, ImageExecutionGraph, ImageGraphInfo, ImageKernel, SampleNode};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::pipeline::ImagePipeline;
use crate::pipeline::op::{
    BatchKernel, CompiledNormalize, CompiledProgram, CompiledSampleOp, ExecutionKind, ImageOp,
    IndexOp, PipelineImageState, SampleKernel, compile_sampler,
};
use crate::pipeline::planning::logical::{FusionGroupPayload, ImagePipelineContext};
use crate::runtime::ImageDataLoader;
use crate::sample::image::ImageAxisOrder;
use crate::sampler::IndexSampler;
use rivet_data::random::{OpKey, RandomContext};
use rivet_exec::physical::{
    ExecutableGraph, ExecutionLane, KernelStage, ParallelismClass, PhysicalGraph, PhysicalLowering,
    PhysicalNodeKind, PhysicalNodeSpec, PhysicalStorageEffect,
};
use rivet_plan::{
    LogicalNode, LogicalPlan, NodeId, NodeKind, PropertyInference, ValueGranularity,
    ValueProperties,
};
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
            crate::pipeline::ImageOptimizationOptions::default(),
        )
    }
    /// Compile with individually controlled logical optimization rules.
    pub fn compile_with_options(
        self,
        options: crate::pipeline::ImageOptimizationOptions,
    ) -> RivetResult<ImageDataLoader> {
        Self::compile_logical_plan_with_options(self.to_logical_plan(), 0, options)
    }
    pub fn compile_logical_plan_with_options(
        plan: LogicalPlan,
        start: usize,
        options: crate::pipeline::ImageOptimizationOptions,
    ) -> RivetResult<ImageDataLoader> {
        compile_graph(plan, start, true, options)
    }
    #[cfg(test)]
    pub(in crate::pipeline) fn compile_unoptimized_for_test(
        self,
        start: usize,
    ) -> RivetResult<ImageDataLoader> {
        compile_graph(
            self.to_logical_plan(),
            start,
            false,
            crate::pipeline::ImageOptimizationOptions::default(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Binding {
    Read,
    Sampler,
    Sink,
    Stack,
    Identity,
    Sample,
    Decode,
    Batch,
    SampleConcat,
    BatchConcat,
    SampleNormalize,
    SampleNormalizeRgb,
    BatchNormalize,
    SampleFused,
    BatchFused,
}

impl Binding {
    const REGISTRY: [(Self, &str); 15] = [
        (Self::Read, "vision.cpu.read"),
        (Self::Sampler, "vision.cpu.sampler"),
        (Self::Sink, "vision.cpu.sink"),
        (Self::Stack, "vision.cpu.stack"),
        (Self::Identity, "vision.cpu.identity"),
        (Self::Sample, "vision.cpu.sample"),
        (Self::Decode, "vision.cpu.decode"),
        (Self::Batch, "vision.cpu.batch"),
        (Self::SampleConcat, "vision.cpu.sample_concat"),
        (Self::BatchConcat, "vision.cpu.batch_concat"),
        (Self::SampleNormalize, "vision.cpu.sample_normalize"),
        (
            Self::SampleNormalizeRgb,
            "vision.cpu.sample_normalize_rgb_table",
        ),
        (Self::BatchNormalize, "vision.cpu.batch_normalize"),
        (Self::SampleFused, "vision.cpu.sample_normalize_to_chw"),
        (Self::BatchFused, "vision.cpu.batch_normalize_to_chw"),
    ];

    fn id(self) -> &'static str {
        Self::REGISTRY[self as usize].1
    }

    fn from_id(id: &str) -> Option<Self> {
        let base = id.split_once('/').map_or(id, |(base, _)| base);
        Self::REGISTRY
            .iter()
            .find_map(|&(binding, name)| (name == base).then_some(binding))
    }
}

fn execution_bindings(
    plan: &LogicalPlan,
    order: &[NodeId],
    annotations: &rivet_plan::PropertyAnnotations,
    workers: usize,
) -> RivetResult<HashMap<NodeId, Binding>> {
    let mut bindings = HashMap::with_capacity(order.len());
    let mut phases = HashMap::<NodeId, (bool, bool)>::with_capacity(order.len());
    for &id in order {
        let node = plan
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let (sample_work, batch_phase) =
            node.inputs()
                .iter()
                .fold((false, false), |(sample, batch), input| {
                    let (input_sample, input_batch) = phases[&input];
                    (sample || input_sample, batch || input_batch)
                });
        let props = node
            .inputs()
            .get(0)
            .and_then(|input| annotations.get(input));
        let input_is_batch = props.is_some_and(|p| p.granularity == Some(ValueGranularity::Batch));
        let sample_normalize = workers > 0 && sample_work && !batch_phase && !input_is_batch;
        let binding = match node.kind() {
            NodeKind::Source => Binding::Read,
            NodeKind::Index => Binding::Sampler,
            NodeKind::Sink => Binding::Sink,
            NodeKind::Batch => Binding::Stack,
            NodeKind::Op if node.payload_as::<crate::pipeline::ImageConcat>().is_some() => {
                if input_is_batch {
                    Binding::BatchConcat
                } else {
                    Binding::SampleConcat
                }
            }
            NodeKind::Op if node.payload_as::<FusionGroupPayload>().is_some() => {
                if sample_normalize {
                    Binding::SampleFused
                } else {
                    Binding::BatchFused
                }
            }
            NodeKind::Op => {
                let op = node.payload_as::<ImageOp>().ok_or_else(|| {
                    invalid_pipeline(format!(
                        "logical node %{} has no registered vision kernel",
                        id.index()
                    ))
                })?;
                match op {
                    ImageOp::Layout(config)
                        if props.is_some_and(|p| {
                            crate::pipeline::planning::inference::state_from_properties(p)
                                .is_ok_and(|state| state_axis_order(state) == config.axis_order)
                        }) =>
                    {
                        Binding::Identity
                    }
                    ImageOp::Decode(_) => Binding::Decode,
                    ImageOp::Normalize(config)
                        if sample_normalize
                            && matches!(config.mean.len(), 1 | 3)
                            && config.mean.iter().zip(&config.std).all(|(&mean, &std)| {
                                (1.0 / (255.0 * std)).is_finite() && (-mean / std).is_finite()
                            })
                            && props.is_some_and(|p| {
                                p.dtype == Some(rivet_plan::DataType::U8)
                                    && p.axis_order == Some(rivet_plan::AxisOrder::Hwc)
                            }) =>
                    {
                        Binding::SampleNormalizeRgb
                    }
                    ImageOp::Normalize(_) if sample_normalize => Binding::SampleNormalize,
                    ImageOp::Normalize(_) => Binding::BatchNormalize,
                    op if op.execution_kind() == ExecutionKind::Batch => Binding::Batch,
                    _ => Binding::Sample,
                }
            }
            kind => {
                return Err(invalid_pipeline(format!(
                    "logical node %{} has no vision kernel for {kind:?}",
                    id.index()
                )));
            }
        };
        phases.insert(
            id,
            (
                sample_work
                    || matches!(
                        binding,
                        Binding::Sample
                            | Binding::Decode
                            | Binding::SampleNormalize
                            | Binding::SampleNormalizeRgb
                            | Binding::SampleFused
                    ),
                batch_phase
                    || matches!(
                        binding,
                        Binding::Stack
                            | Binding::Batch
                            | Binding::BatchConcat
                            | Binding::BatchNormalize
                            | Binding::BatchFused
                    ),
            ),
        );
        bindings.insert(id, binding);
    }
    Ok(bindings)
}

pub(in crate::pipeline) fn register_execution_kernels(
    caps: &mut rivet_plan::KernelCapabilities,
    plan: &LogicalPlan,
    annotations: &rivet_plan::PropertyAnnotations,
    workers: usize,
) -> RivetResult<()> {
    use rivet_plan::{AxisOrder, KernelClass};
    let order = plan
        .topological_order()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let bindings = execution_bindings(plan, &order, annotations, workers)?;
    for id in order {
        let binding = bindings[&id];
        let candidates = caps
            .for_node(plan, id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let desired_class = match binding {
            Binding::Read | Binding::Sampler => KernelClass::Source,
            Binding::Sink => KernelClass::Sink,
            Binding::Stack | Binding::Batch | Binding::BatchConcat | Binding::BatchNormalize => {
                KernelClass::Batch
            }
            Binding::SampleFused | Binding::BatchFused => KernelClass::Fused,
            _ => KernelClass::Sample,
        };
        let mut capability = candidates
            .into_iter()
            .find(|kernel| {
                kernel.class == desired_class && kernel.device == rivet_plan::DeviceClass::Cpu
            })
            .cloned()
            .ok_or_else(|| {
                invalid_pipeline(format!(
                    "logical node %{}: no registration for {}",
                    id.index(),
                    binding.id()
                ))
            })?;
        capability.name = binding.id().to_owned();
        // Stage requirements in the semantic registry describe logical values.
        // Here the selected implementation may subsequently lift across stack.
        capability.requirements.output_stage = None;
        if desired_class == KernelClass::Fused
            && annotations.get(id).and_then(|p| p.granularity) == Some(ValueGranularity::Batch)
        {
            capability.requirements.input_granularity = Some(ValueGranularity::Batch);
            capability.requirements.output_granularity = Some(ValueGranularity::Batch);
            capability.requirements.input_axis_order = Some(AxisOrder::Nhwc);
            capability.requirements.output_axis_order = Some(AxisOrder::Nchw);
        }
        if binding == Binding::SampleNormalizeRgb {
            capability.requirements.input_axis_order = Some(AxisOrder::Hwc);
            capability.requirements.input_dtype = Some(rivet_plan::DataType::U8);
            capability.requirements.input_granularity = Some(ValueGranularity::Sample);
        }
        if matches!(binding, Binding::SampleNormalize | Binding::BatchNormalize) {
            for (axis, suffix) in [
                (AxisOrder::Hwc, "hwc"),
                (AxisOrder::Chw, "chw"),
                (AxisOrder::Nhwc, "nhwc"),
                (AxisOrder::Nchw, "nchw"),
            ] {
                let mut layout = capability.clone();
                layout.name = format!("{}/{suffix}", binding.id());
                layout.requirements.input_axis_order = Some(axis);
                caps.register_for_node(id, layout);
            }
        } else {
            caps.register_for_node(id, capability);
        }
    }
    Ok(())
}

struct VisionPhysicalLowering<'a> {
    kernels: &'a HashMap<NodeId, (PhysicalNodeKind, Arc<ImageKernel>)>,
    selected: &'a HashMap<NodeId, &'a rivet_plan::PlacementCandidate>,
}
impl PhysicalLowering for VisionPhysicalLowering<'_> {
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
        spec.storage_effect = match operator.as_ref() {
            ImageKernel::Concat { .. } => PhysicalStorageEffect::Copy,
            ImageKernel::Sample(node)
                if node.op.is_decode()
                    || matches!(
                        node.op.kernel,
                        SampleKernel::SampleNormalize(_)
                            | SampleKernel::SampleNormalizeRgb(..)
                            | SampleKernel::SampleNormalizeToChw(_)
                    ) =>
            {
                PhysicalStorageEffect::Materialize
            }
            ImageKernel::BatchOp(node) if matches!(node.op, BatchKernel::Layout { .. }) => {
                PhysicalStorageEffect::View
            }
            ImageKernel::BatchOp(node)
                if matches!(
                    node.op,
                    BatchKernel::Normalize(_) | BatchKernel::NormalizeToChw(_)
                ) =>
            {
                PhysicalStorageEffect::Materialize
            }
            _ => spec.storage_effect,
        };
        // These typed kernels act independently on every image. Lifting
        // their contiguous suffix across stacking preserves image order and
        // padding semantics; stage classification alone grants no permission.
        if matches!(operator.as_ref(), ImageKernel::BatchOp(node) if matches!(node.op,
            BatchKernel::Normalize(_) | BatchKernel::NormalizeToChw(_) |
            BatchKernel::ConvertImageDtype { .. } | BatchKernel::ToDtype { .. } | BatchKernel::Layout { .. }))
        {
            spec = spec.with_batch_lift_equivalence();
        }
        if let Some(candidate) = self.selected.get(&id) {
            spec.selection = Some((*candidate).clone());
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
pub(in crate::pipeline) fn select_terminal_normalize_layout(
    plan: &mut LogicalPlan,
    annotations: &rivet_plan::PropertyAnnotations,
    workers: usize,
) -> RivetResult<Option<&'static str>> {
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
            return Ok(None);
        }
    }
    let Some((&sink, prefix)) = order.split_last() else {
        return Ok(None);
    };
    if plan
        .node(sink)
        .map_err(|error| invalid_pipeline(error.to_string()))?
        .kind()
        != NodeKind::Sink
    {
        return Ok(None);
    }
    let Some((&batch, prefix)) = prefix.split_last() else {
        return Ok(None);
    };
    if plan
        .node(batch)
        .map_err(|error| invalid_pipeline(error.to_string()))?
        .kind()
        != NodeKind::Batch
    {
        return Ok(None);
    }
    let Some((&group_id, preceding)) = prefix.split_last() else {
        return Ok(None);
    };
    let group_node = plan
        .node(group_id)
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let Some(group) = group_node.payload_as::<FusionGroupPayload>() else {
        return Ok(None);
    };
    let [ImageOp::Normalize(normalize), ImageOp::Layout(layout)] = group.ops.as_slice() else {
        return Ok(None);
    };
    if group.name != "NormalizeToChw" {
        return Ok(None);
    }
    let Some(input) = group_node.inputs().get(0) else {
        return Ok(None);
    };
    let Some(properties) = annotations.get(input) else {
        return Ok(None);
    };
    if properties.granularity != Some(ValueGranularity::Sample) {
        return Ok(None);
    }
    let has_sample_work = properties
        .operator
        .is_some_and(|operator| operator.sample_stage_has_work);
    let source_supports_batch_read = preceding.iter().any(|&id| {
        plan.node(id)
            .ok()
            .and_then(|node| node.payload_as::<crate::pipeline::op::SourceOp>())
            .is_some_and(|source| source.supports_batch_read())
    });
    let reason = if workers > 0 && has_sample_work {
        "preserve worker-local sample normalization and a terminal layout view"
    } else if !has_sample_work && source_supports_batch_read {
        "preserve batch-native reads and three-channel normalization; layout is a view"
    } else {
        return Ok(None);
    };
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
            return Ok(None);
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
    Ok(Some(reason))
}

fn compile_graph(
    mut logical: LogicalPlan,
    start: usize,
    optimize: bool,
    options: crate::pipeline::ImageOptimizationOptions,
) -> RivetResult<ImageDataLoader> {
    crate::pipeline::planning::logical::assign_missing_random_identities(&mut logical)?;
    logical
        .canonicalize()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let context = logical
        .context_as::<ImagePipelineContext>()
        .ok_or_else(|| invalid_pipeline("logical plan has no vision context"))?
        .clone();
    if !context.batch_mix_valid {
        return Err(invalid_pipeline(
            "MixUp/CutMix must follow an explicit batch and all image/index operations",
        ));
    }
    for config in &context.batch_mixing {
        config.validate()?;
    }
    if let Some((id, node)) = logical
        .nodes()
        .find(|(_, node)| node.kind() == NodeKind::DeviceCut)
    {
        let target = node
            .payload_as::<rivet_plan::DeviceCut>()
            .map(|cut| &cut.target);
        return Err(invalid_pipeline(format!(
            "vision DeviceCut execution is not implemented yet at logical node %{} for {target:?}",
            id.index(),
        )));
    }
    let (annotations, placement, mut diagnostics) = if optimize {
        let (mut optimizer_context, placement) =
            crate::pipeline::planning::optimizer::optimize_vision_plan_for_execution(
                &mut logical,
                context.runtime.num_workers,
                options,
            )?;
        (
            optimizer_context.annotations().clone(),
            Some(placement),
            std::mem::take(&mut optimizer_context.diagnostics),
        )
    } else {
        (
            logical
                .infer_properties(&crate::pipeline::planning::inference::VisionPropertyInference)
                .map_err(crate::pipeline::planning::logical::inference_error)?,
            None,
            Vec::new(),
        )
    };
    if !annotations.is_valid_for(&logical) {
        return Err(invalid_pipeline(
            "compile requires properties for the current plan revision",
        ));
    }
    if let Some(placement) = &placement
        && !placement.is_valid_for(&logical)
    {
        let selected = placement
            .candidates
            .first()
            .map(|candidate| {
                format!(
                    "logical node %{} candidate {}",
                    candidate.node.index(),
                    candidate.kernel
                )
            })
            .unwrap_or_else(|| "empty placement".to_owned());
        return Err(invalid_pipeline(format!(
            "{selected} is from placement revision {:?}; current revision {:?}",
            placement.revision,
            logical.revision(),
        )));
    }
    let order = logical
        .topological_order()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let sources = order
        .iter()
        .filter_map(|id| {
            logical
                .node(*id)
                .ok()?
                .payload_as::<crate::pipeline::op::SourceOp>()
        })
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
                .payload_as::<crate::pipeline::op::BatchConfig>()
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
            if annotations.get(id).is_some_and(|properties| {
                crate::pipeline::planning::inference::views_per_source(properties) > 1
            }) {
                return Err(invalid_pipeline(
                    "skip/take/shuffle after multi-view crops is not supported; place source sampling before five_crop/ten_crop",
                ));
            }
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
    let fallback = placement
        .is_none()
        .then(|| execution_bindings(&logical, &order, &annotations, context.runtime.num_workers))
        .transpose()?;
    let selected: HashMap<_, _> = placement
        .iter()
        .flat_map(|plan| &plan.candidates)
        .map(|candidate| (candidate.node, candidate))
        .collect();
    for &id in &order {
        let node = logical
            .node(id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let candidate = selected.get(&id).copied();
        let binding = if let Some(candidate) = candidate {
            let invalid_binding = || {
                invalid_pipeline(format!(
                    "logical node %{} cannot bind candidate {} on {:?}",
                    id.index(),
                    candidate.kernel,
                    candidate.device,
                ))
            };
            if candidate.device != rivet_plan::DeviceClass::Cpu {
                return Err(invalid_binding());
            }
            let binding = Binding::from_id(&candidate.kernel).ok_or_else(invalid_binding)?;
            let compatible = match binding {
                Binding::Read => node.kind() == NodeKind::Source,
                Binding::Sampler => node.kind() == NodeKind::Index,
                Binding::Sink => node.kind() == NodeKind::Sink,
                Binding::Stack => node.kind() == NodeKind::Batch,
                Binding::SampleConcat | Binding::BatchConcat => {
                    node.payload_as::<crate::pipeline::ImageConcat>().is_some()
                }
                Binding::SampleFused | Binding::BatchFused => {
                    node.payload_as::<FusionGroupPayload>().is_some()
                }
                Binding::Identity => {
                    matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Layout(_)))
                }
                Binding::Decode => matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Decode(_))),
                Binding::SampleNormalize
                | Binding::SampleNormalizeRgb
                | Binding::BatchNormalize => {
                    matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Normalize(_)))
                }
                Binding::Sample | Binding::Batch => node.payload_as::<ImageOp>().is_some(),
            };
            if !compatible {
                return Err(invalid_binding());
            }
            if matches!(binding, Binding::SampleNormalize | Binding::BatchNormalize) {
                let axis = node
                    .inputs()
                    .get(0)
                    .and_then(|input| annotations.get(input))
                    .and_then(|p| p.axis_order.as_ref());
                let suffix = match axis {
                    Some(rivet_plan::AxisOrder::Hwc) => "hwc",
                    Some(rivet_plan::AxisOrder::Chw) => "chw",
                    Some(rivet_plan::AxisOrder::Nhwc) => "nhwc",
                    Some(rivet_plan::AxisOrder::Nchw) => "nchw",
                    _ => return Err(invalid_binding()),
                };
                if candidate.kernel.split_once('/') != Some((binding.id(), suffix)) {
                    return Err(invalid_binding());
                }
            } else if candidate.kernel != binding.id() {
                return Err(invalid_binding());
            }
            binding
        } else {
            fallback
                .as_ref()
                .map(|bindings| bindings[&id])
                .ok_or_else(|| {
                    invalid_pipeline(format!(
                        "logical node %{} has no selected implementation",
                        id.index()
                    ))
                })?
        };
        let input_state = node
            .inputs()
            .get(0)
            .and_then(|input| annotations.get(input))
            .map(crate::pipeline::planning::inference::state_from_properties)
            .transpose()?
            .unwrap_or(source.state());
        let (kind, kernel) = match binding {
            Binding::Read => (
                PhysicalNodeKind::Source,
                ImageKernel::Source(source.clone()),
            ),
            Binding::Sampler => (PhysicalNodeKind::Sampler, ImageKernel::Identity),
            Binding::Sink => (PhysicalNodeKind::Sink, ImageKernel::Identity),
            Binding::Stack => (
                PhysicalNodeKind::Batch,
                ImageKernel::Batch {
                    axis_order: state_axis_order(input_state),
                },
            ),
            _ => {
                if let Some(concat) = node.payload_as::<crate::pipeline::ImageConcat>() {
                    let stage = if binding == Binding::BatchConcat {
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
                    if binding == Binding::SampleFused {
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
                    if binding == Binding::Identity {
                        (
                            PhysicalNodeKind::Kernel(KernelStage::Sample),
                            ImageKernel::Identity,
                        )
                    } else if matches!(binding, Binding::Batch | Binding::BatchNormalize) {
                        let input_layout = state_axis_order(input_state);
                        let kernel = match op {
                            ImageOp::V2(crate::transforms::V2Config::ToDtype { dtype, scale }) => {
                                BatchKernel::ToDtype {
                                    dtype,
                                    scale,
                                    input_layout,
                                }
                            }
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
                        let output_layout = state_axis_order(
                            crate::pipeline::planning::inference::state_from_properties(
                                annotations.get(id).expect("inferred output"),
                            )?,
                        );
                        batch_ops.push(kernel.clone());
                        (
                            PhysicalNodeKind::Kernel(KernelStage::Batch),
                            ImageKernel::BatchOp(Arc::new(BatchNode {
                                op: kernel,
                                output_layout,
                            })),
                        )
                    } else {
                        let kernel_stage = if binding == Binding::Decode {
                            KernelStage::Decode
                        } else {
                            KernelStage::Sample
                        };
                        let compiled = if let ImageOp::Normalize(config) = op {
                            let compiled = CompiledNormalize {
                                config,
                                input_layout: state_axis_order(input_state),
                            };
                            CompiledSampleOp {
                                kernel: if binding == Binding::SampleNormalizeRgb {
                                    compiled.into_rgb_sample_kernel()
                                } else {
                                    SampleKernel::SampleNormalize(compiled)
                                },
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
        };
        kernels.insert(id, (kind, Arc::new(kernel)));
    }
    let mut physical = PhysicalGraph::lower(
        &logical,
        &VisionPhysicalLowering {
            kernels: &kernels,
            selected: &selected,
        },
    )
    .map_err(|error| invalid_pipeline(error.to_string()))?;
    physical
        .ensure_sampler_before_source()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let mut sample_nodes = Vec::new();
    let mut batch_nodes = Vec::new();
    let physical_order = physical
        .execution_order()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let inference = crate::pipeline::planning::inference::VisionPropertyInference;
    let mut edge_values = vec![None::<ValueProperties>; physical.nodes().len()];
    let mut pre_batch_state = None;
    let mut managed_bytes = context.runtime.dag_managed_budget_bytes.map(|_| 0usize);
    for id in &physical_order {
        let node = physical
            .node(*id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        if node.kind == PhysicalNodeKind::Sampler {
            // Reordering places index operations on the request stream.
            continue;
        }
        let logical_id = node.logical_id.ok_or_else(|| {
            invalid_pipeline(format!(
                "physical node p{} has no vision binding",
                id.index()
            ))
        })?;
        let logical_node = logical
            .node(logical_id)
            .map_err(|error| invalid_pipeline(error.to_string()))?;
        let inputs = if node.kind == PhysicalNodeKind::Source {
            Vec::new()
        } else {
            node.inputs
                .iter()
                .map(|input| {
                    edge_values[input.index()].clone().ok_or_else(|| {
                        invalid_pipeline(format!(
                            "missing image properties on edge p{} → p{}",
                            input.index(),
                            id.index(),
                        ))
                    })
                })
                .collect::<RivetResult<Vec<_>>>()?
        };
        let output = inference
            .infer_node(logical_node, &inputs)
            .map_err(|error| {
                invalid_pipeline(format!(
                    "physical inputs of p{} (logical %{}): {error}",
                    id.index(),
                    logical_id.index(),
                ))
            })?;
        let (_, kernel) = &kernels[&logical_id];
        if context.runtime.dag_managed_budget_bytes.is_some() {
            let footprint = match kernel.as_ref() {
                ImageKernel::Sample(node) if node.op.is_decode() => Some((0, &output)),
                ImageKernel::Batch { .. } => Some((std::mem::size_of::<i64>(), &inputs[0])),
                _ => None,
            };
            if let Some((extra, props)) = footprint {
                managed_bytes = managed_bytes.and_then(|current| {
                    let per_image = known_image_bytes(props)?.checked_add(extra)?;
                    let views = crate::pipeline::planning::inference::views_per_source(props);
                    let items = batch.size.checked_mul(views)?;
                    current.checked_add(per_image.checked_mul(items)?)
                });
            }
        }
        edge_values[id.index()] = Some(output);
        match kernel.as_ref() {
            ImageKernel::Sample(node) => sample_nodes.push(Arc::clone(node)),
            ImageKernel::BatchOp(node) => batch_nodes.push(Arc::clone(node)),
            ImageKernel::Batch { .. } => {
                // Lifting changes the stack's producer and its layout.
                let state =
                    crate::pipeline::planning::inference::state_from_properties(&inputs[0])?;
                pre_batch_state.get_or_insert(state);
                physical
                    .set_operator(
                        *id,
                        Arc::new(ImageKernel::Batch {
                            axis_order: state_axis_order(state),
                        }),
                    )
                    .map_err(|error| invalid_pipeline(error.to_string()))?;
            }
            _ => {}
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
        && !sample_ops.iter().any(|op| op.output_multiplier() != 1)
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
    if !linear {
        let reason = if !linear_topology {
            "shared or multi-input physical graph"
        } else if sample_ops.iter().any(|op| op.output_multiplier() != 1) {
            "sample kernel expands cardinality"
        } else {
            "typed linear execution cannot preserve physical stage order"
        };
        diagnostics.push(rivet_plan::Diagnostic {
            code: "execution.dag",
            message: reason.to_owned(),
        });
    }
    physical.set_diagnostics(diagnostics);
    let root = logical
        .root()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let output_state = crate::pipeline::planning::inference::state_from_properties(
        annotations.get(root).expect("inferred root"),
    )?;
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
    let physical_root = physical
        .root()
        .map_err(|error| invalid_pipeline(error.to_string()))?;
    let physical_output = edge_values[physical_root.index()]
        .as_ref()
        .ok_or_else(|| invalid_pipeline("physical sink has no image properties"))?;
    let logical_output = annotations.get(root).expect("inferred root");
    if physical_output.granularity != logical_output.granularity
        || physical_output.residency != logical_output.residency
        || physical_output.axis_order != logical_output.axis_order
        || physical_output.dtype != logical_output.dtype
        || physical_output.shape != logical_output.shape
        || crate::pipeline::planning::inference::views_per_source(physical_output)
            != crate::pipeline::planning::inference::views_per_source(logical_output)
    {
        return Err(invalid_pipeline(
            "physical sink output violates the logical image contract",
        ));
    }
    let pre_batch_state = pre_batch_state.expect("batch barrier compiled");
    let info = ImageGraphInfo {
        input_state: source.state(),
        views_per_sample: crate::pipeline::planning::inference::views_per_source(
            annotations.get(root).expect("inferred root"),
        ),
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
        managed_bytes_per_morsel: managed_bytes,
        explanation,
    });
    ImageDataLoader::new(graph, IndexSampler::new(sampler, start), context.runtime)
        .map(|loader| loader.with_batch_mixing(context.batch_mixing, random))
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
        ImageOp::RandomChoice { choices, weights } => {
            let cumulative_weights =
                crate::pipeline::op::choice_cumulative_weights(choices.len(), weights.as_deref())?;
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
                SampleKernel::RandomChoice {
                    key,
                    branches,
                    cumulative_weights,
                },
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

fn known_image_bytes(properties: &ValueProperties) -> Option<usize> {
    let PipelineImageState::Decoded { dtype, .. } =
        crate::pipeline::planning::inference::state_from_properties(properties).ok()?
    else {
        return None;
    };
    properties
        .shape
        .as_ref()?
        .dims()
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dim| match dim {
            rivet_plan::ShapeDim::Known(size) => bytes.checked_mul(*size),
            rivet_plan::ShapeDim::Dynamic => None,
        })
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

    fn decoded_pipeline() -> ImagePipeline {
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
    }

    fn pipeline(workers: usize) -> ImagePipeline {
        decoded_pipeline()
            .convert_image_dtype(DType::U8)
            .crop(1, 1, 2, 2)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .workers(workers)
            .batch(2, false)
    }

    #[test]
    fn loader_exposes_the_executed_graph_as_structured_facts() {
        let mut loader = pipeline(2).compile().unwrap();
        let explanation = loader.physical_explanation().unwrap();
        assert_eq!(explanation.nodes.last().unwrap().id, explanation.root);
        let text = loader.physical_explain().unwrap();
        assert!(text.starts_with(&explanation.to_string()));
        let mut seen = std::collections::HashSet::new();
        for node in &explanation.nodes {
            assert!(node.inputs.iter().all(|input| seen.contains(input)));
            assert_eq!(node.logical_id.is_some(), node.operator.is_some());
            seen.insert(node.id);
        }
    }

    #[test]
    fn terminal_normalize_layout_retains_the_batch_native_fast_path() {
        let configure = || {
            decoded_pipeline()
                .normalize(vec![0.5; 3], vec![0.5; 3])
                .workers(4)
        };
        let mut selected = configure().hwc_to_chw().batch(2, false).compile().unwrap();
        let mut split = configure().batch(2, false).hwc_to_chw().compile().unwrap();
        assert!(selected.info.can_use_batch_native());
        assert_eq!(selected.info.first_batch_op_name(), Some("Normalize"));
        let explain = selected.physical_explanation().unwrap();
        assert!(
            explain
                .diagnostics
                .iter()
                .any(|entry| entry.code == "fusion.execution-split")
        );
        let selected = selected.next_batch().unwrap().unwrap();
        let split = split.next_batch().unwrap().unwrap();
        assert_eq!(
            selected.images.to_vec::<f32>().unwrap(),
            split.images.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            selected.labels.to_vec::<i64>().unwrap(),
            split.labels.to_vec::<i64>().unwrap()
        );
    }

    #[test]
    fn normalize_layout_selects_and_runs_the_bound_implementation() {
        for (axis, expected_suffix, expected_dims) in [
            (ImageAxisOrder::Hwc, "hwc", [2, 4, 4, 3]),
            (ImageAxisOrder::Chw, "chw", [2, 3, 4, 4]),
        ] {
            let pipeline = decoded_pipeline();
            let pipeline = match axis {
                ImageAxisOrder::Hwc => pipeline,
                ImageAxisOrder::Chw => pipeline.hwc_to_chw(),
            }
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .workers(0)
            .batch(2, false);
            let mut logical = pipeline.to_logical_plan();
            let (_, placement) =
                crate::pipeline::planning::optimizer::optimize_vision_plan_for_execution(
                    &mut logical,
                    0,
                    crate::pipeline::ImageOptimizationOptions::default(),
                )
                .unwrap();
            let expected = format!("vision.cpu.batch_normalize/{expected_suffix}");
            assert!(
                placement
                    .candidates
                    .iter()
                    .any(|candidate| candidate.kernel == expected)
            );
            let mut loader = pipeline.compile().unwrap();
            let explanation = loader.physical_explanation().unwrap();
            let node = explanation
                .nodes
                .iter()
                .find(|node| {
                    node.selection
                        .as_ref()
                        .is_some_and(|candidate| candidate.kernel == expected)
                })
                .unwrap();
            let candidate = node.selection.as_ref().unwrap();
            assert_eq!(candidate.rejected.len(), 3);
            assert!(candidate.rejected.iter().all(|rejection| {
                rejection.reason == rivet_plan::KernelRejectionReason::ValueProperties
            }));
            assert_eq!(node.operator.as_deref(), Some("Normalize"));
            assert!(loader.physical_explain().unwrap().contains(&expected));
            let batch = loader.next_batch().unwrap().unwrap();
            assert_eq!(batch.images.dims(), expected_dims);
            assert_eq!(batch.labels.to_vec::<i64>().unwrap(), [3, 7]);
            let values = batch.images.to_vec::<f32>().unwrap();
            for (position, value) in values.into_iter().enumerate() {
                let index = match axis {
                    ImageAxisOrder::Hwc => position,
                    ImageAxisOrder::Chw => {
                        position / 48 * 48 + position % 16 * 3 + position / 16 % 3
                    }
                };
                assert!((value - (index as f32 / 255.0 - 0.5) / 0.5).abs() < 1e-6);
            }
            assert!(loader.next_batch().unwrap().is_none());
        }
    }

    #[test]
    fn execution_preparation_places_once_and_keeps_semantic_explain_separate() {
        let mut declared = pipeline(2).to_logical_plan();
        let (context, placement) =
            crate::pipeline::planning::optimizer::optimize_vision_plan_for_execution(
                &mut declared,
                2,
                crate::pipeline::ImageOptimizationOptions::default(),
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
        let (context, _) = crate::pipeline::planning::optimizer::optimize_vision_plan_with_options(
            &mut semantic,
            2,
            crate::pipeline::ImageOptimizationOptions::default(),
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
    fn nonfinite_normalization_coefficients_keep_the_affine_binding() {
        let mut loader = decoded_pipeline()
            .crop(1, 1, 2, 2)
            .normalize(vec![f32::INFINITY; 3], vec![0.5; 3])
            .workers(2)
            .batch(2, false)
            .compile()
            .unwrap();
        let explanation = loader.physical_explanation().unwrap();
        assert!(explanation.nodes.iter().any(|node| {
            node.selection
                .as_ref()
                .is_some_and(|candidate| candidate.kernel == "vision.cpu.sample_normalize/hwc")
        }));
        let batch = loader.next_batch().unwrap().unwrap();
        assert!(
            batch
                .images
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&value| { value == f32::NEG_INFINITY })
        );
    }

    #[test]
    fn compiled_rgb_table_preserves_affine_bits_for_every_byte_and_view() {
        use crate::pipeline::op::SampleContext;
        use crate::sample::image::{DecodedSample, ImageSample};

        let image = Tensor::from_vec(
            (0..1536)
                .map(|i| ((i / 3 + 53 * (i % 3)) % 256) as u8)
                .collect(),
            [2, 256, 3],
            &Device::Cpu,
        )
        .unwrap();
        let state = PipelineImageState::Decoded {
            dtype: DType::U8,
            axis_order: ImageAxisOrder::Hwc,
        };
        for config in [
            crate::transforms::NormalizeConfig::new(vec![0.43], vec![0.29]),
            crate::transforms::NormalizeConfig::new(vec![0.49, 0.48, 0.45], vec![0.25, 0.24, 0.26]),
        ] {
            let compiled = CompiledSampleOp {
                kernel: CompiledNormalize {
                    config: config.clone(),
                    input_layout: ImageAxisOrder::Hwc,
                }
                .into_rgb_sample_kernel(),
                random_key: None,
                input_state: state,
            };
            for view in [
                image.clone(),
                image.narrow(0, 1, 1).unwrap(),
                image.narrow(1, 2, 200).unwrap(),
            ] {
                let input = ImageSample::Decoded(DecodedSample {
                    image: view,
                    label: 7,
                });
                let expected = config
                    .apply_trusted(input.clone(), ImageAxisOrder::Hwc)
                    .unwrap()
                    .into_decoded()
                    .unwrap();
                let actual = compiled
                    .execute(input, &SampleContext::new(0))
                    .unwrap()
                    .into_decoded()
                    .unwrap();
                assert_eq!(actual.label, expected.label);
                assert_eq!(actual.image.dims(), expected.image.dims());
                let bits = |tensor: &Tensor| {
                    tensor
                        .to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&actual.image), bits(&expected.image));
            }
        }
    }

    #[test]
    fn identity_conversion_crop_and_fused_normalize_preserve_values() {
        for workers in [0, 2] {
            let mut optimized = pipeline(workers).compile().unwrap();
            if workers > 0 {
                let graph = optimized.physical_explanation().unwrap();
                assert!(graph.nodes.iter().any(|node| {
                    node.storage_effect == PhysicalStorageEffect::Materialize
                        && node.selection.as_ref().is_some_and(|candidate| {
                            candidate.kernel == "vision.cpu.sample_normalize_rgb_table"
                        })
                }));
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
