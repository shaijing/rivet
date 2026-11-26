//! Domain-neutral physical graph and morsel execution protocol.
//!
//! Logical plans describe semantic operations. This module lowers their
//! topology to executable physical nodes without putting backend handles in
//! `rivet-plan`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rivet_core::{DType, Tensor};
use rivet_plan::{DeviceClass, LogicalNode, LogicalPlan, NodeId, PlanError, ValueGranularity};

use crate::memory::KernelMemoryRequirements;
use crate::runtime::{RuntimeError, RuntimeResult};

/// Stable index into a [`PhysicalGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysNodeId(usize);

impl PhysNodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Runtime lane selected during physical planning. Device handles and streams
/// remain runtime-owned and are deliberately absent from this description.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionLane {
    Io,
    Cpu,
    Transfer,
    Device { class: DeviceClass, ordinal: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelStage {
    Decode,
    Sample,
    Batch,
}

/// Where a physical kernel exposes parallel work. This lets the scheduler
/// distinguish parallelizing samples from parallelizing one batch or device
/// invocation, and helps avoid nested CPU oversubscription.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ParallelismClass {
    #[default]
    Serial,
    AcrossSamples,
    WithinBatch,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// Physical operation category. Domain-specific operation details are held by
/// the associated type-erased operator, when present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicalNodeKind {
    Sampler,
    Source,
    Kernel(KernelStage),
    Batch,
    Transfer(TransferKind),
    Cache,
    Sink,
}

impl fmt::Display for PhysicalNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sampler => f.write_str("Sampler"),
            Self::Source => f.write_str("Source"),
            Self::Kernel(KernelStage::Decode) => f.write_str("Decode"),
            Self::Kernel(KernelStage::Sample) => f.write_str("SampleKernel"),
            Self::Kernel(KernelStage::Batch) => f.write_str("BatchKernel"),
            Self::Batch => f.write_str("Batch"),
            Self::Transfer(TransferKind::HostToDevice) => f.write_str("Transfer(H2D)"),
            Self::Transfer(TransferKind::DeviceToHost) => f.write_str("Transfer(D2H)"),
            Self::Transfer(TransferKind::DeviceToDevice) => f.write_str("Transfer(D2D)"),
            Self::Cache => f.write_str("Cache"),
            Self::Sink => f.write_str("Sink"),
        }
    }
}

/// A type-erased domain value. Encoded and decoded values are held in shared
/// ownership so fan-out does not require copying their backing buffers.
pub type TensorBackingVisitor = fn(&dyn Any, &mut dyn FnMut(&Tensor));

#[derive(Clone)]
pub enum ExecutionValue {
    Encoded(Arc<dyn Any + Send + Sync>),
    Decoded(Arc<dyn Any + Send + Sync>),
    DecodedWithTensors(Arc<dyn Any + Send + Sync>, TensorBackingVisitor),
    Tensor(Tensor),
}

impl fmt::Debug for ExecutionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoded(_) => f.write_str("Encoded(..)"),
            Self::Decoded(_) | Self::DecodedWithTensors(..) => f.write_str("Decoded(..)"),
            Self::Tensor(tensor) => f.debug_tuple("Tensor").field(tensor).finish(),
        }
    }
}

/// Unit of data flowing between physical stages.
#[derive(Clone, Debug)]
pub struct Morsel {
    pub sequence_id: u64,
    pub source_indices: Vec<usize>,
    pub values: Vec<ExecutionValue>,
    /// Legacy producer-reported logical payload weight. Zero may mean unreported;
    /// never interpret this field as backing allocation capacity or RSS.
    pub bytes: usize,
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<DType>,
    pub residency: MorselResidency,
    pub granularity: ValueGranularity,
}

impl Morsel {
    pub fn new(sequence_id: u64, source_indices: Vec<usize>) -> Self {
        Self {
            sequence_id,
            source_indices,
            values: Vec::new(),
            bytes: 0,
            shape: None,
            dtype: None,
            residency: MorselResidency::Unknown,
            granularity: ValueGranularity::Sample,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MorselResidency {
    Host,
    Device { class: DeviceClass, ordinal: usize },
    Unknown,
}

/// Domain kernel or IO operation attached to a physical node.
pub trait PhysicalOperator: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, morsel: Morsel) -> RuntimeResult<Morsel>;

    /// Opt in when unary execution is equivalent to execute_inputs.
    fn supports_direct_unary(&self) -> bool {
        false
    }

    /// Optional workspace for a directly invoked unary kernel. The mutable
    /// workspace borrow is confined to this call; outputs must own storage.
    fn workspace_type(&self) -> Option<TypeId> {
        None
    }

    fn new_workspace(&self) -> Box<dyn Any + Send + Sync> {
        Box::new(())
    }

    fn execute_in_workspace(
        &self,
        input: Morsel,
        _workspace: &mut (dyn Any + Send + Sync),
    ) -> RuntimeResult<Morsel> {
        self.execute(input)
    }

    /// Addressable output bytes allocated by a managed kernel. Returning
    /// None leaves allocations by this operator outside managed accounting.
    fn managed_output_bytes(&self, _inputs: &[Morsel]) -> RuntimeResult<Option<usize>> {
        Ok(None)
    }

    /// Report the backing created by a kernel with a managed output bound.
    /// Opaque envelopes can override this to expose their tensors.
    fn visit_managed_output(&self, output: &Morsel, visit: &mut dyn FnMut(&Tensor)) {
        for value in &output.values {
            match value {
                ExecutionValue::Tensor(tensor) => visit(tensor),
                ExecutionValue::DecodedWithTensors(value, tensors) => {
                    tensors(value.as_ref(), visit)
                }
                _ => {}
            }
        }
    }

    /// Ordered inputs for a join. Unary operators reject unexpected arity.
    fn execute_inputs(&self, mut inputs: Vec<Morsel>) -> RuntimeResult<Morsel> {
        if inputs.len() != 1 {
            return Err(RuntimeError::Message(format!(
                "{} requires one input, got {}",
                self.name(),
                inputs.len()
            )));
        }
        self.execute(inputs.pop().expect("checked unary input"))
    }
}

/// An output boundary with a known storage effect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhysicalStorageEffect {
    #[default]
    Unspecified,
    SourceRead,
    View,
    Materialize,
    Copy,
    Transfer,
}

/// Output for one logical node during lowering. A lowerer may fuse multiple
/// logical nodes in a later extension; the initial contract maps one logical
/// node to one physical node.
pub struct PhysicalNodeSpec {
    pub kind: PhysicalNodeKind,
    pub lane: ExecutionLane,
    pub operator: Option<Arc<dyn PhysicalOperator>>,
    pub selection: Option<rivet_plan::PlacementCandidate>,
    pub storage_effect: PhysicalStorageEffect,
    /// Destination lane for a transfer node. Transfer execution itself runs on
    /// the transfer lane; the target remains explicit for planning/runtime.
    pub transfer_target: Option<ExecutionLane>,
    /// Estimated payload size from placement cost analysis, when known.
    pub estimated_transfer_bytes: Option<u64>,
    pub parallelism: ParallelismClass,
    pub memory_requirements: KernelMemoryRequirements,
    pub estimated_output_bytes: Option<usize>,
    /// Domain proof that applying this unary kernel per sample is equivalent
    /// to applying it after stacking. Stage classification alone is insufficient.
    pub batch_lift_equivalent: bool,
}

impl PhysicalNodeSpec {
    pub fn new(kind: PhysicalNodeKind, lane: ExecutionLane) -> Self {
        Self {
            kind,
            lane,
            operator: None,
            selection: None,
            storage_effect: match kind {
                PhysicalNodeKind::Source => PhysicalStorageEffect::SourceRead,
                PhysicalNodeKind::Batch => PhysicalStorageEffect::Materialize,
                PhysicalNodeKind::Transfer(_) => PhysicalStorageEffect::Transfer,
                _ => PhysicalStorageEffect::Unspecified,
            },
            transfer_target: None,
            estimated_transfer_bytes: None,
            parallelism: ParallelismClass::Serial,
            memory_requirements: KernelMemoryRequirements::default(),
            estimated_output_bytes: None,
            batch_lift_equivalent: false,
        }
    }

    /// Authorize lifting across a stack barrier only; never across another kernel.
    pub fn with_batch_lift_equivalence(mut self) -> Self {
        self.batch_lift_equivalent = true;
        self
    }

    pub fn with_operator(mut self, operator: Arc<dyn PhysicalOperator>) -> Self {
        self.operator = Some(operator);
        self
    }

    pub fn with_transfer_target(mut self, target: ExecutionLane) -> Self {
        self.transfer_target = Some(target);
        self
    }

    pub fn with_estimated_transfer_bytes(mut self, bytes: u64) -> Self {
        self.estimated_transfer_bytes = Some(bytes);
        self
    }

    pub fn with_parallelism(mut self, parallelism: ParallelismClass) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn with_memory_requirements(mut self, requirements: KernelMemoryRequirements) -> Self {
        self.memory_requirements = requirements;
        self
    }

    pub fn with_estimated_output_bytes(mut self, bytes: usize) -> Self {
        self.estimated_output_bytes = Some(bytes);
        self
    }
}

/// Contract for domain-owned LogicalPlan → PhysicalGraph lowering.
pub trait PhysicalLowering {
    fn lower_node(
        &self,
        logical_id: NodeId,
        logical_node: &LogicalNode,
        physical_inputs: &[PhysNodeId],
    ) -> RuntimeResult<PhysicalNodeSpec>;
}

#[derive(Clone)]
pub struct PhysicalNode {
    pub id: PhysNodeId,
    pub logical_id: Option<NodeId>,
    pub kind: PhysicalNodeKind,
    pub lane: ExecutionLane,
    pub inputs: Vec<PhysNodeId>,
    /// Number of graph edges that consume this node's result, plus one for a
    /// root result retained by the caller.
    pub last_use_count: usize,
    pub operator: Option<Arc<dyn PhysicalOperator>>,
    pub selection: Option<rivet_plan::PlacementCandidate>,
    pub storage_effect: PhysicalStorageEffect,
    pub transfer_target: Option<ExecutionLane>,
    pub estimated_transfer_bytes: Option<u64>,
    pub parallelism: ParallelismClass,
    pub memory_requirements: KernelMemoryRequirements,
    pub estimated_output_bytes: Option<usize>,
    /// Domain proof that applying this unary kernel per sample is equivalent
    /// to applying it after stacking. Stage classification alone is insufficient.
    pub batch_lift_equivalent: bool,
}

/// Runtime graph facts in execution order, including synthetic nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalExplanation {
    pub root: PhysNodeId,
    pub nodes: Vec<PhysicalNodeExplanation>,
    pub diagnostics: Vec<rivet_plan::Diagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalNodeExplanation {
    pub id: PhysNodeId,
    pub logical_id: Option<NodeId>,
    pub kind: PhysicalNodeKind,
    pub lane: ExecutionLane,
    pub inputs: Vec<PhysNodeId>,
    pub last_use_count: usize,
    pub operator: Option<String>,
    pub selection: Option<rivet_plan::PlacementCandidate>,
    pub storage_effect: PhysicalStorageEffect,
    pub transfer_target: Option<ExecutionLane>,
    pub estimated_transfer_bytes: Option<u64>,
    pub parallelism: ParallelismClass,
}

impl fmt::Display for PhysicalExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PhysicalGraph(root=p{})", self.root.index())?;
        for node in &self.nodes {
            let inputs = node
                .inputs
                .iter()
                .map(|id| format!("p{}", id.index()))
                .collect::<Vec<_>>()
                .join(", ");
            let logical = node
                .logical_id
                .map(|id| format!(" logical=%{}", id.index()))
                .unwrap_or_default();
            let operator = node
                .operator
                .as_ref()
                .map(|op| format!(" op={op}"))
                .unwrap_or_default();
            let transfer_target = node
                .transfer_target
                .as_ref()
                .map(|target| format!(" target={target:?}"))
                .unwrap_or_default();
            let transfer_estimate = node
                .estimated_transfer_bytes
                .map(|bytes| format!(" estimated_transfer_bytes={bytes}"))
                .unwrap_or_default();
            let implementation = node
                .selection
                .as_ref()
                .map(|candidate| {
                    format!(
                        " kernel={} cost={:.6}s candidates={:?} rejected={:?}",
                        candidate.kernel,
                        candidate.cost.total_score,
                        candidate.alternatives,
                        candidate.rejected,
                    )
                })
                .unwrap_or_default();
            writeln!(
                f,
                "  p{} {} lane={:?} parallelism={:?}{}{}{}{}{} <- [{}] last_uses={} storage={:?}",
                node.id.index(),
                node.kind,
                node.lane,
                node.parallelism,
                logical,
                operator,
                implementation,
                transfer_target,
                transfer_estimate,
                inputs,
                node.last_use_count,
                node.storage_effect,
            )?;
        }
        for diagnostic in &self.diagnostics {
            writeln!(f, "  {}: {}", diagnostic.code, diagnostic.message)?;
        }
        Ok(())
    }
}

impl fmt::Debug for PhysicalNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicalNode")
            .field("id", &self.id)
            .field("logical_id", &self.logical_id)
            .field("kind", &self.kind)
            .field("lane", &self.lane)
            .field("inputs", &self.inputs)
            .field("last_use_count", &self.last_use_count)
            .field("operator", &self.operator.as_ref().map(|op| op.name()))
            .field("transfer_target", &self.transfer_target)
            .field("estimated_transfer_bytes", &self.estimated_transfer_bytes)
            .field("parallelism", &self.parallelism)
            .field("memory_requirements", &self.memory_requirements)
            .field("estimated_output_bytes", &self.estimated_output_bytes)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PhysicalGraphError {
    #[error("DAG execution cancelled")]
    Cancelled,
    #[error(
        "node p{node} reported logical output bytes exceed single-value limit after allocation: {bytes} > {limit}; not a queue limit or managed executor budget"
    )]
    ValueLogicalLimit {
        node: usize,
        bytes: usize,
        limit: usize,
    },
    #[error("node p{node}: {source}")]
    MemoryAccounting {
        node: usize,
        #[source]
        source: crate::memory::MemoryAccountingError,
    },
    #[error(transparent)]
    Logical(#[from] PlanError),
    #[error("physical graph has no root")]
    MissingRoot,
    #[error("physical node id {0} is outside the graph")]
    InvalidNode(usize),
    #[error("physical graph contains a cycle through node {0}")]
    Cycle(usize),
    #[error("unsupported physical graph execution: {0}")]
    UnsupportedExecutionShape(String),
    #[error("physical operator failed: {0}")]
    Operator(#[from] RuntimeError),
}

/// Owned physical graph. `last_use_count` is computed when validating or
/// lowering the graph and can drive buffer release decisions in later stages.
#[derive(Clone, Default)]
pub struct PhysicalGraph {
    nodes: Vec<PhysicalNode>,
    root: Option<PhysNodeId>,
    diagnostics: Vec<rivet_plan::Diagnostic>,
}

impl PhysicalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_diagnostics(&mut self, diagnostics: Vec<rivet_plan::Diagnostic>) {
        self.diagnostics = diagnostics;
    }

    pub fn add_node(
        &mut self,
        logical_id: Option<NodeId>,
        spec: PhysicalNodeSpec,
        inputs: impl IntoIterator<Item = PhysNodeId>,
    ) -> PhysNodeId {
        let id = PhysNodeId(self.nodes.len());
        self.nodes.push(PhysicalNode {
            id,
            logical_id,
            kind: spec.kind,
            lane: spec.lane,
            inputs: inputs.into_iter().collect(),
            last_use_count: 0,
            operator: spec.operator,
            selection: spec.selection,
            storage_effect: spec.storage_effect,
            transfer_target: spec.transfer_target,
            estimated_transfer_bytes: spec.estimated_transfer_bytes,
            parallelism: spec.parallelism,
            memory_requirements: spec.memory_requirements,
            estimated_output_bytes: spec.estimated_output_bytes,
            batch_lift_equivalent: spec.batch_lift_equivalent,
        });
        id
    }

    pub fn set_root(&mut self, root: PhysNodeId) -> Result<(), PhysicalGraphError> {
        self.node(root)?;
        self.root = Some(root);
        Ok(())
    }

    pub fn root(&self) -> Result<PhysNodeId, PhysicalGraphError> {
        self.root.ok_or(PhysicalGraphError::MissingRoot)
    }

    pub fn nodes(&self) -> &[PhysicalNode] {
        &self.nodes
    }

    pub fn node(&self, id: PhysNodeId) -> Result<&PhysicalNode, PhysicalGraphError> {
        self.nodes
            .get(id.0)
            .ok_or(PhysicalGraphError::InvalidNode(id.0))
    }

    pub fn set_operator(
        &mut self,
        id: PhysNodeId,
        operator: Arc<dyn PhysicalOperator>,
    ) -> Result<(), PhysicalGraphError> {
        self.nodes
            .get_mut(id.index())
            .ok_or(PhysicalGraphError::InvalidNode(id.index()))?
            .operator = Some(operator);
        Ok(())
    }

    pub fn execution_order(&self) -> Result<Vec<PhysNodeId>, PhysicalGraphError> {
        self.topological_order()
    }

    /// Add explicit transfer nodes for host/device lane changes on the
    /// input edges, including DAG branches. Host I/O and CPU share residency.
    pub fn insert_transfers_for_lane_changes(
        &mut self,
    ) -> Result<Vec<PhysNodeId>, PhysicalGraphError> {
        let order = self.topological_order()?;
        let original_len = self.nodes.len();
        let mut inserted = Vec::new();
        for consumer in order {
            if consumer.index() >= original_len {
                continue;
            }
            let consumer_lane = self.node(consumer)?.lane.clone();
            let inputs = self.node(consumer)?.inputs.clone();
            for (input_position, input) in inputs.into_iter().enumerate() {
                let producer_lane = output_lane(self.node(input)?);
                let Some(kind) = transfer_for_lanes(&producer_lane, &consumer_lane) else {
                    continue;
                };
                let transfer = self.add_node(
                    None,
                    PhysicalNodeSpec::new(
                        PhysicalNodeKind::Transfer(kind),
                        ExecutionLane::Transfer,
                    )
                    .with_transfer_target(consumer_lane.clone()),
                    [input],
                );
                self.nodes[consumer.index()].inputs[input_position] = transfer;
                inserted.push(transfer);
            }
        }
        self.validate()?;
        Ok(inserted)
    }

    pub fn set_transfer_estimate(
        &mut self,
        id: PhysNodeId,
        bytes: u64,
    ) -> Result<(), PhysicalGraphError> {
        let node = self
            .nodes
            .get_mut(id.index())
            .ok_or(PhysicalGraphError::InvalidNode(id.index()))?;
        if !matches!(node.kind, PhysicalNodeKind::Transfer(_)) {
            return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                "p{} is not a transfer node",
                id.index()
            )));
        }
        node.estimated_transfer_bytes = Some(bytes);
        Ok(())
    }

    /// Materialize an implicit sequential sampler before source reads when a
    /// logical plan has no explicit index operation nodes.
    pub fn ensure_sampler_before_source(&mut self) -> Result<(), PhysicalGraphError> {
        if self
            .nodes
            .iter()
            .any(|node| node.kind == PhysicalNodeKind::Sampler)
        {
            return Ok(());
        }
        let source = self
            .nodes
            .iter()
            .find(|node| node.kind == PhysicalNodeKind::Source)
            .map(|node| node.id)
            .ok_or_else(|| {
                PhysicalGraphError::UnsupportedExecutionShape(
                    "cannot insert sampler because graph has no source".to_string(),
                )
            })?;
        let inputs = std::mem::take(&mut self.nodes[source.0].inputs);
        let sampler = self.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sampler, ExecutionLane::Cpu),
            inputs,
        );
        self.nodes[source.0].inputs = vec![sampler];
        self.validate()?;
        Ok(())
    }

    pub fn lower(
        logical: &LogicalPlan,
        lowerer: &dyn PhysicalLowering,
    ) -> Result<Self, PhysicalLoweringError> {
        logical.validate()?;
        let mut graph = Self::new();
        let mut ids = HashMap::<NodeId, PhysNodeId>::new();
        for logical_id in logical.topological_order()? {
            let logical_node = logical.node(logical_id)?;
            let inputs = logical
                .parents(logical_id)?
                .into_iter()
                .map(|id| {
                    ids.get(&id)
                        .copied()
                        .ok_or(PhysicalLoweringError::MissingLoweredInput(id.index()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let spec = lowerer
                .lower_node(logical_id, logical_node, &inputs)
                .map_err(PhysicalLoweringError::Runtime)?;
            let physical_id = graph.add_node(Some(logical_id), spec, inputs);
            ids.insert(logical_id, physical_id);
        }
        let root = logical.root()?;
        graph.set_root(
            *ids.get(&root)
                .ok_or(PhysicalLoweringError::MissingLoweredInput(root.index()))?,
        )?;
        graph.move_samplers_before_source()?;
        graph.move_batch_kernels_after_batch()?;
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&mut self) -> Result<(), PhysicalGraphError> {
        let root = self.root()?;
        self.node(root)?;
        let mut consumers = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            match node.kind {
                PhysicalNodeKind::Transfer(_) => {
                    if node.lane != ExecutionLane::Transfer
                        || node.transfer_target.is_none()
                        || node.inputs.len() != 1
                    {
                        return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                            "p{} transfer requires one input, a Transfer lane, and an explicit target",
                            node.id.index()
                        )));
                    }
                    let source_lane = output_lane(self.node(node.inputs[0])?);
                    let expected = transfer_for_lanes(
                        &source_lane,
                        node.transfer_target
                            .as_ref()
                            .expect("checked transfer target"),
                    );
                    if expected
                        != Some(match node.kind {
                            PhysicalNodeKind::Transfer(kind) => kind,
                            _ => unreachable!(),
                        })
                    {
                        return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                            "p{} transfer kind does not match its source and target lanes",
                            node.id.index()
                        )));
                    }
                }
                _ if node.transfer_target.is_some() => {
                    return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                        "p{} has a transfer target but is not a transfer node",
                        node.id.index()
                    )));
                }
                _ if node.estimated_transfer_bytes.is_some() => {
                    return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                        "p{} has a transfer estimate but is not a transfer node",
                        node.id.index()
                    )));
                }
                _ => {}
            }
            for input in &node.inputs {
                let producer = self.node(*input)?;
                if let PhysicalNodeKind::Transfer(_) = producer.kind {
                    let target = producer
                        .transfer_target
                        .as_ref()
                        .expect("validated transfer target");
                    if !lanes_match(target, &node.lane) {
                        return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                            "transfer p{} targets {target:?}, but consumer p{} runs on {:?}",
                            producer.id.index(),
                            node.id.index(),
                            node.lane
                        )));
                    }
                }
                consumers[input.0] += 1;
            }
        }
        consumers[root.0] += 1;
        for (node, uses) in self.nodes.iter_mut().zip(consumers) {
            node.last_use_count = uses;
        }
        self.topological_order()?;
        Ok(())
    }

    pub fn explain(&mut self) -> Result<String, PhysicalGraphError> {
        self.validate()?;
        Ok(self.explanation(&self.topological_order()?)?.to_string())
    }

    fn explanation(&self, order: &[PhysNodeId]) -> Result<PhysicalExplanation, PhysicalGraphError> {
        let nodes = order
            .iter()
            .map(|&id| {
                let node = self.node(id)?;
                Ok(PhysicalNodeExplanation {
                    id,
                    logical_id: node.logical_id,
                    kind: node.kind,
                    lane: node.lane.clone(),
                    inputs: node.inputs.clone(),
                    last_use_count: node.last_use_count,
                    operator: node.operator.as_ref().map(|op| op.name().to_owned()),
                    selection: node.selection.clone(),
                    storage_effect: node.storage_effect,
                    transfer_target: node.transfer_target.clone(),
                    estimated_transfer_bytes: node.estimated_transfer_bytes,
                    parallelism: node.parallelism,
                })
            })
            .collect::<Result<_, PhysicalGraphError>>()?;
        Ok(PhysicalExplanation {
            root: self.root()?,
            nodes,
            diagnostics: self.diagnostics.clone(),
        })
    }

    fn topological_order(&self) -> Result<Vec<PhysNodeId>, PhysicalGraphError> {
        let mut state = vec![0u8; self.nodes.len()];
        fn visit(
            id: PhysNodeId,
            nodes: &[PhysicalNode],
            state: &mut [u8],
            out: &mut Vec<PhysNodeId>,
        ) -> Result<(), PhysicalGraphError> {
            if id.0 >= nodes.len() {
                return Err(PhysicalGraphError::InvalidNode(id.0));
            }
            match state[id.0] {
                1 => return Err(PhysicalGraphError::Cycle(id.0)),
                2 => return Ok(()),
                _ => {}
            }
            state[id.0] = 1;
            for input in &nodes[id.0].inputs {
                visit(*input, nodes, state, out)?;
            }
            state[id.0] = 2;
            out.push(id);
            Ok(())
        }
        let mut out = Vec::new();
        visit(self.root()?, &self.nodes, &mut state, &mut out)?;
        if out.len() != self.nodes.len() {
            let missing = state.iter().position(|state| *state == 0).unwrap_or(0);
            return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                "node p{missing} is not reachable from the root"
            )));
        }
        Ok(out)
    }

    /// Lift only an explicitly equivalent contiguous suffix across stacking.
    /// Kernel stage describes execution granularity, not semantic commutativity.
    fn move_batch_kernels_after_batch(&mut self) -> Result<(), PhysicalGraphError> {
        let order = self.topological_order()?;
        if self.nodes.iter().any(|node| node.inputs.len() > 1) {
            return Ok(());
        }
        let mut consumers = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            for input in &node.inputs {
                consumers[input.0] += 1;
            }
        }
        if consumers.iter().any(|count| *count > 1) {
            return Ok(());
        }
        let batches = order
            .iter()
            .enumerate()
            .filter_map(|(index, id)| {
                (self.nodes[id.0].kind == PhysicalNodeKind::Batch).then_some(index)
            })
            .collect::<Vec<_>>();
        if batches.len() != 1 {
            return Ok(());
        }
        let batch_index = batches[0];
        let batch = &self.nodes[order[batch_index].0];
        if batch.lane != ExecutionLane::Cpu || batch.inputs.len() != 1 {
            return Ok(());
        }
        let suffix_start = order[..batch_index]
            .iter()
            .rposition(|id| {
                let node = &self.nodes[id.0];
                node.kind != PhysicalNodeKind::Kernel(KernelStage::Batch)
                    || !node.batch_lift_equivalent
                    || node.inputs.len() != 1
                    || node.lane != batch.lane
            })
            .map_or(0, |position| position + 1);
        let moved = order[suffix_start..batch_index].to_vec();
        if moved.is_empty() {
            return Ok(());
        }

        let mut reordered = order[..suffix_start].to_vec();
        reordered.push(order[batch_index]);
        reordered.extend(moved);
        reordered.extend(order[batch_index + 1..].iter().copied());

        for (position, id) in reordered.iter().copied().enumerate() {
            self.nodes[id.0].inputs = if position == 0 {
                Vec::new()
            } else {
                vec![reordered[position - 1]]
            };
        }
        Ok(())
    }

    fn move_samplers_before_source(&mut self) -> Result<(), PhysicalGraphError> {
        let sources = self
            .nodes
            .iter()
            .filter(|node| node.kind == PhysicalNodeKind::Source)
            .map(|node| node.id)
            .collect::<Vec<_>>();
        for source in sources {
            let mut prefix = Vec::new();
            let mut current = source;
            loop {
                let consumers = self
                    .nodes
                    .iter()
                    .filter(|node| node.inputs.contains(&current))
                    .map(|node| node.id)
                    .collect::<Vec<_>>();
                let [consumer] = consumers.as_slice() else {
                    break;
                };
                if self.nodes[consumer.0].kind != PhysicalNodeKind::Sampler
                    || self.nodes[consumer.0].inputs.len() != 1
                {
                    break;
                }
                prefix.push(*consumer);
                current = *consumer;
            }
            if prefix.is_empty() {
                continue;
            }
            let first = prefix[0];
            let last = *prefix.last().expect("nonempty prefix");
            // Replace uses of the sampler prefix with the source. Preserve
            // every branch edge and keep sampling before physical source I/O.
            for node in &mut self.nodes {
                if node.id == source || prefix.contains(&node.id) {
                    continue;
                }
                for input in &mut node.inputs {
                    if *input == last {
                        *input = source;
                    }
                }
            }
            self.nodes[first.0].inputs = std::mem::take(&mut self.nodes[source.0].inputs);
            self.nodes[source.0].inputs = vec![last];
            if self.root == Some(last) {
                self.root = Some(source);
            }
        }
        Ok(())
    }
}

fn transfer_for_lanes(from: &ExecutionLane, to: &ExecutionLane) -> Option<TransferKind> {
    match (from, to) {
        (ExecutionLane::Cpu | ExecutionLane::Io, ExecutionLane::Device { .. }) => {
            Some(TransferKind::HostToDevice)
        }
        (ExecutionLane::Device { .. }, ExecutionLane::Cpu | ExecutionLane::Io) => {
            Some(TransferKind::DeviceToHost)
        }
        (
            ExecutionLane::Device {
                class: from_class,
                ordinal: from,
            },
            ExecutionLane::Device {
                class: to_class,
                ordinal: to,
            },
        ) if from != to || from_class != to_class => Some(TransferKind::DeviceToDevice),
        _ => None,
    }
}

fn output_lane(node: &PhysicalNode) -> ExecutionLane {
    node.transfer_target
        .clone()
        .unwrap_or_else(|| node.lane.clone())
}

fn lanes_match(lhs: &ExecutionLane, rhs: &ExecutionLane) -> bool {
    match (lhs, rhs) {
        (ExecutionLane::Cpu | ExecutionLane::Io, ExecutionLane::Cpu | ExecutionLane::Io) => true,
        (
            ExecutionLane::Device {
                class: lhs_class,
                ordinal: lhs,
            },
            ExecutionLane::Device {
                class: rhs_class,
                ordinal: rhs,
            },
        ) => lhs == rhs && lhs_class == rhs_class,
        _ => false,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PhysicalLoweringError {
    #[error(transparent)]
    Logical(#[from] PlanError),
    #[error(transparent)]
    Graph(#[from] PhysicalGraphError),
    #[error("logical input node %{0} has not been lowered")]
    MissingLoweredInput(usize),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeProfile {
    pub executions: u64,
    /// Time spent executing this physical node's action.
    pub elapsed: Duration,
    /// Time spent blocked on adjacent bounded stage queues.
    pub wait_elapsed: Duration,
    pub wait_events: u64,
    /// Cumulative producer-reported logical bytes per execution/port.
    /// Shared values may be counted repeatedly; zero may be unreported.
    pub input_bytes: u64,
    /// Cumulative reported logical output bytes, not live backing capacity.
    pub output_bytes: u64,
}

#[derive(Default)]
struct AtomicNodeProfile {
    executions: AtomicU64,
    elapsed_ns: AtomicU64,
    wait_ns: AtomicU64,
    wait_events: AtomicU64,
    input_bytes: AtomicU64,
    output_bytes: AtomicU64,
}

#[derive(Default)]
struct AtomicMemoryProfile {
    boundary_observations: AtomicU64,
    peak_backing: AtomicU64,
    peak_unknown: AtomicU64,
    incomplete: AtomicU64,
    value_failures: AtomicU64,
    live_failures: AtomicU64,
    scheduler: Option<AtomicDagSchedulerProfile>,
}

/// Timing and depth observed at the DAG coordinator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DagSchedulerProfile {
    /// Time waiting for a result when the worker result channel was empty.
    pub result_wait: Duration,
    /// Time a completed result spent buffered behind an earlier sequence.
    pub reorder_wait: Duration,
    /// Coordinator time blocked on capacity until a delivery releases it.
    pub budget_wait: Duration,
    /// Empty-channel wait with no later result buffered.
    pub worker_wait: Duration,
    /// Empty-channel wait while a later result is buffered.
    pub earliest_sequence_wait: Duration,
    /// Time between delivery and the next pull with outstanding work.
    pub downstream_wait: Duration,
    pub preflight_time: Duration,
    pub preflight_morsels: u64,
    pub preflight_source_reads: u64,
    pub preflight_source_read_time: Duration,
    pub preflight_dimension_checks: u64,
    pub peak_pending_morsels: u64,
    pub peak_pending_encoded_bytes: u64,
    pub peak_reordered_logical_bytes: u64,
    pub peak_reordered_reserved_bytes: u64,
    pub peak_in_flight_morsels: u64,
    pub peak_reordered_morsels: u64,
}

#[derive(Default)]
struct AtomicDagSchedulerProfile {
    result_wait_ns: AtomicU64,
    reorder_wait_ns: AtomicU64,
    budget_wait_ns: AtomicU64,
    worker_wait_ns: AtomicU64,
    earliest_sequence_wait_ns: AtomicU64,
    downstream_wait_ns: AtomicU64,
    preflight_ns: AtomicU64,
    preflight_morsels: AtomicU64,
    preflight_source_reads: AtomicU64,
    preflight_source_read_ns: AtomicU64,
    preflight_dimension_checks: AtomicU64,
    peak_pending_morsels: AtomicU64,
    peak_pending_encoded_bytes: AtomicU64,
    peak_reordered_logical_bytes: AtomicU64,
    peak_reordered_reserved_bytes: AtomicU64,
    peak_in_flight: AtomicU64,
    peak_reordered: AtomicU64,
}

fn add_duration(counter: &AtomicU64, duration: Duration) {
    counter.fetch_add(
        duration.as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
}

#[derive(Clone)]
pub struct PhysicalProfiler {
    nodes: Option<Arc<[AtomicNodeProfile]>>,
    memory: Option<Arc<AtomicMemoryProfile>>,
}

impl PhysicalProfiler {
    fn scheduler(&self) -> Option<&AtomicDagSchedulerProfile> {
        self.memory.as_ref()?.scheduler.as_ref()
    }

    pub fn dag_scheduler_snapshot(&self) -> Option<DagSchedulerProfile> {
        let scheduler = self.scheduler()?;
        Some(DagSchedulerProfile {
            result_wait: Duration::from_nanos(scheduler.result_wait_ns.load(Ordering::Relaxed)),
            reorder_wait: Duration::from_nanos(scheduler.reorder_wait_ns.load(Ordering::Relaxed)),
            budget_wait: Duration::from_nanos(scheduler.budget_wait_ns.load(Ordering::Relaxed)),
            worker_wait: Duration::from_nanos(scheduler.worker_wait_ns.load(Ordering::Relaxed)),
            earliest_sequence_wait: Duration::from_nanos(
                scheduler.earliest_sequence_wait_ns.load(Ordering::Relaxed),
            ),
            downstream_wait: Duration::from_nanos(
                scheduler.downstream_wait_ns.load(Ordering::Relaxed),
            ),
            preflight_time: Duration::from_nanos(scheduler.preflight_ns.load(Ordering::Relaxed)),
            preflight_morsels: scheduler.preflight_morsels.load(Ordering::Relaxed),
            preflight_source_reads: scheduler.preflight_source_reads.load(Ordering::Relaxed),
            preflight_source_read_time: Duration::from_nanos(
                scheduler.preflight_source_read_ns.load(Ordering::Relaxed),
            ),
            preflight_dimension_checks: scheduler
                .preflight_dimension_checks
                .load(Ordering::Relaxed),
            peak_pending_morsels: scheduler.peak_pending_morsels.load(Ordering::Relaxed),
            peak_pending_encoded_bytes: scheduler
                .peak_pending_encoded_bytes
                .load(Ordering::Relaxed),
            peak_reordered_logical_bytes: scheduler
                .peak_reordered_logical_bytes
                .load(Ordering::Relaxed),
            peak_reordered_reserved_bytes: scheduler
                .peak_reordered_reserved_bytes
                .load(Ordering::Relaxed),
            peak_in_flight_morsels: scheduler.peak_in_flight.load(Ordering::Relaxed),
            peak_reordered_morsels: scheduler.peak_reordered.load(Ordering::Relaxed),
        })
    }

    pub(crate) fn observe_dag_depth(&self, in_flight: u64, reordered: usize) {
        if let Some(scheduler) = self.scheduler() {
            scheduler
                .peak_in_flight
                .fetch_max(in_flight, Ordering::Relaxed);
            scheduler
                .peak_reordered
                .fetch_max(reordered as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_dag_reorder_residence(&self, elapsed: Duration) {
        if let Some(scheduler) = self.scheduler() {
            add_duration(&scheduler.reorder_wait_ns, elapsed);
        }
    }

    pub fn record_dag_preflight_read(&self, elapsed: Duration) {
        if let Some(scheduler) = self.scheduler() {
            scheduler
                .preflight_source_reads
                .fetch_add(1, Ordering::Relaxed);
            add_duration(&scheduler.preflight_source_read_ns, elapsed);
        }
    }

    pub fn record_dag_preflight_dimension_checks(&self, count: usize) {
        if let Some(scheduler) = self.scheduler() {
            scheduler
                .preflight_dimension_checks
                .fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn observe_dag_preflight(&self, elapsed: Duration, bytes: Option<usize>) {
        if let Some(scheduler) = self.scheduler() {
            scheduler.preflight_morsels.fetch_add(1, Ordering::Relaxed);
            add_duration(&scheduler.preflight_ns, elapsed);
            if let Some(bytes) = bytes {
                scheduler
                    .peak_pending_morsels
                    .fetch_max(1, Ordering::Relaxed);
                scheduler
                    .peak_pending_encoded_bytes
                    .fetch_max(bytes as u64, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn observe_dag_reorder_bytes(&self, logical: u64, reserved: usize) {
        if let Some(scheduler) = self.scheduler() {
            scheduler
                .peak_reordered_logical_bytes
                .fetch_max(logical, Ordering::Relaxed);
            scheduler
                .peak_reordered_reserved_bytes
                .fetch_max(reserved as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_dag_worker_wait(&self, elapsed: Duration, waiting_for_sequence: bool) {
        if let Some(scheduler) = self.scheduler() {
            add_duration(&scheduler.result_wait_ns, elapsed);
            let counter = if waiting_for_sequence {
                &scheduler.earliest_sequence_wait_ns
            } else {
                &scheduler.worker_wait_ns
            };
            add_duration(counter, elapsed);
        }
    }

    pub(crate) fn record_dag_budget_wait(&self, elapsed: Duration) {
        if let Some(scheduler) = self.scheduler() {
            add_duration(&scheduler.budget_wait_ns, elapsed);
        }
    }

    pub(crate) fn record_dag_downstream_wait(&self, elapsed: Duration) {
        if let Some(scheduler) = self.scheduler() {
            add_duration(&scheduler.downstream_wait_ns, elapsed);
        }
    }

    /// Graph boundary observations only; stage queues retain their own reported
    /// payload counters. Managed executor budgeting and kernel scratch are absent.
    pub fn memory_snapshot(&self) -> Option<crate::memory::GraphMemoryProfile> {
        let memory = self.memory.as_ref()?;
        Some(crate::memory::GraphMemoryProfile {
            boundary_observations: memory.boundary_observations.load(Ordering::Relaxed),
            peak_known_backing_bytes_per_invocation: memory.peak_backing.load(Ordering::Relaxed),
            peak_unknown_values_per_invocation: memory.peak_unknown.load(Ordering::Relaxed),
            incomplete_observations: memory.incomplete.load(Ordering::Relaxed),
            value_logical_limit_failures: memory.value_failures.load(Ordering::Relaxed),
            observed_live_limit_failures: memory.live_failures.load(Ordering::Relaxed),
        })
    }
    fn record_memory(&self, snapshot: crate::memory::LiveMemorySnapshot) {
        let Some(memory) = &self.memory else {
            return;
        };
        memory.boundary_observations.fetch_add(1, Ordering::Relaxed);
        memory
            .peak_backing
            .fetch_max(snapshot.known_backing_bytes as u64, Ordering::Relaxed);
        memory
            .peak_unknown
            .fetch_max(snapshot.unknown_values as u64, Ordering::Relaxed);
        if !snapshot.is_complete() {
            memory.incomplete.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn record_memory_failure(&self, live: bool) {
        if let Some(memory) = &self.memory {
            let counter = if live {
                &memory.live_failures
            } else {
                &memory.value_failures
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Create an enabled profiler with one dense counter slot per physical node.
    /// Profiling is opt-in because even atomic accounting perturbs short kernels.
    pub fn enabled(node_count: usize) -> Self {
        Self::enabled_with_memory(node_count, AtomicMemoryProfile::default())
    }

    pub(crate) fn enabled_for_dag(node_count: usize) -> Self {
        Self::enabled_with_memory(
            node_count,
            AtomicMemoryProfile {
                scheduler: Some(AtomicDagSchedulerProfile::default()),
                ..Default::default()
            },
        )
    }

    fn enabled_with_memory(node_count: usize, memory: AtomicMemoryProfile) -> Self {
        let nodes = (0..node_count)
            .map(|_| AtomicNodeProfile::default())
            .collect::<Vec<_>>()
            .into();
        Self {
            nodes: Some(nodes),
            memory: Some(Arc::new(memory)),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.nodes.is_some()
    }

    pub fn snapshot(&self) -> HashMap<PhysNodeId, NodeProfile> {
        let Some(nodes) = &self.nodes else {
            return HashMap::new();
        };
        nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let profile = NodeProfile {
                    executions: node.executions.load(Ordering::Relaxed),
                    elapsed: Duration::from_nanos(node.elapsed_ns.load(Ordering::Relaxed)),
                    wait_elapsed: Duration::from_nanos(node.wait_ns.load(Ordering::Relaxed)),
                    wait_events: node.wait_events.load(Ordering::Relaxed),
                    input_bytes: node.input_bytes.load(Ordering::Relaxed),
                    output_bytes: node.output_bytes.load(Ordering::Relaxed),
                };
                (profile.executions != 0 || profile.wait_events != 0)
                    .then_some((PhysNodeId(index), profile))
            })
            .collect()
    }

    pub(crate) fn record(&self, id: PhysNodeId, elapsed: Duration, input: usize, output: usize) {
        let Some(profile) = self.nodes.as_ref().and_then(|nodes| nodes.get(id.index())) else {
            return;
        };
        profile.executions.fetch_add(1, Ordering::Relaxed);
        profile.elapsed_ns.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        profile
            .input_bytes
            .fetch_add(input as u64, Ordering::Relaxed);
        profile
            .output_bytes
            .fetch_add(output as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_wait(&self, id: PhysNodeId, elapsed: Duration) {
        if elapsed.is_zero() {
            return;
        }
        let Some(profile) = self.nodes.as_ref().and_then(|nodes| nodes.get(id.index())) else {
            return;
        };
        profile.wait_ns.fetch_add(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        profile.wait_events.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for PhysicalProfiler {
    fn default() -> Self {
        Self {
            nodes: None,
            memory: None,
        }
    }
}

/// Executes attached operators in topological order. Fan-out shares the
/// morsel's cloneable values; domain operators that mutate backing storage
/// must use copy-on-write or produce a fresh output.
/// Validated graph with cached scheduling and edge liveness. Each invocation
/// owns its outputs, so one prepared graph can execute concurrently.
pub struct ExecutableGraph {
    graph: PhysicalGraph,
    order: Vec<PhysNodeId>,
    schedule: ExecutionSchedule,
}

impl ExecutableGraph {
    pub fn new(mut graph: PhysicalGraph) -> Result<Self, PhysicalGraphError> {
        graph.validate()?;
        let order = graph.execution_order()?;
        let schedule = ExecutionSchedule::new(&graph, &order)?;
        Ok(Self {
            graph,
            order,
            schedule,
        })
    }

    pub fn execute_with_limit(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
        max_value_logical_bytes: usize,
    ) -> Result<Morsel, PhysicalGraphError> {
        self.execute_with_memory_limits(
            morsel,
            profiler,
            crate::memory::GraphMemoryLimits {
                max_value_logical_bytes,
                ..Default::default()
            },
        )
    }

    /// Explicit per-value and observed per-invocation limits. Checks happen at
    /// node boundaries AFTER allocation; this is not shared budget admission.
    pub fn execute_with_memory_limits(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
        limits: crate::memory::GraphMemoryLimits,
    ) -> Result<Morsel, PhysicalGraphError> {
        self.execute_reusing(morsel, profiler, limits, &mut self.scratch())
    }

    pub fn explanation(&self) -> Result<PhysicalExplanation, PhysicalGraphError> {
        self.graph.explanation(&self.order)
    }

    pub fn graph(&self) -> &PhysicalGraph {
        &self.graph
    }

    pub fn execute(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
    ) -> Result<Morsel, PhysicalGraphError> {
        self.execute_with_memory_limits(morsel, profiler, Default::default())
    }

    pub(crate) fn scratch(&self) -> GraphScratch {
        GraphScratch::new(self.graph.nodes.len(), self.schedule.workspace_count)
    }

    pub(crate) fn execute_reusing(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
        limits: crate::memory::GraphMemoryLimits,
        scratch: &mut GraphScratch,
    ) -> Result<Morsel, PhysicalGraphError> {
        self.execute_reusing_with_budget(morsel, profiler, limits, scratch, None, None)
    }

    pub(crate) fn execute_reusing_with_budget(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
        limits: crate::memory::GraphMemoryLimits,
        scratch: &mut GraphScratch,
        reservation: Option<&mut crate::memory::MemoryReservation>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Morsel, PhysicalGraphError> {
        execute_graph(
            &self.graph,
            &self.schedule,
            morsel,
            profiler,
            limits,
            scratch,
            reservation,
            cancelled,
        )
    }
}

pub struct GraphExecutor;

impl GraphExecutor {
    pub fn execute(
        graph: &mut PhysicalGraph,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
    ) -> Result<Morsel, PhysicalGraphError> {
        graph.validate()?;
        let schedule = ExecutionSchedule::new(graph, &graph.execution_order()?)?;
        execute_graph(
            graph,
            &schedule,
            morsel,
            profiler,
            Default::default(),
            &mut GraphScratch::new(graph.nodes.len(), schedule.workspace_count),
            None,
            None,
        )
    }
}

struct ScheduledNode {
    id: PhysNodeId,
    input_slots: Vec<usize>,
    direct_unary: bool,
    workspace: Option<usize>,
}

struct ExecutionSchedule {
    root: usize,
    nodes: Vec<ScheduledNode>,
    initial_uses: Vec<usize>,
    entry_uses: usize,
    workspace_count: usize,
}

impl ExecutionSchedule {
    fn new(graph: &PhysicalGraph, order: &[PhysNodeId]) -> Result<Self, PhysicalGraphError> {
        for id in order {
            let node = graph.node(*id)?;
            if matches!(node.kind, PhysicalNodeKind::Transfer(_)) {
                if node.operator.is_none() {
                    return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                        "transfer p{} has no bound operator",
                        id.index(),
                    )));
                }
                continue;
            }
            for input in &node.inputs {
                let producer = graph.node(*input)?;
                if transfer_for_lanes(&output_lane(producer), &node.lane).is_some() {
                    return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                        "edge p{} → p{} needs an explicit bound transfer",
                        input.index(),
                        id.index(),
                    )));
                }
            }
        }
        let workspace_slots =
            crate::memory::MemoryPlanner::workspace_slots(order.iter().filter_map(|&id| {
                let node = &graph.nodes[id.index()];
                let operator = node.operator.as_ref()?;
                if node.inputs.len() > 1 || !operator.supports_direct_unary() {
                    return None;
                }
                Some((id, operator.workspace_type()?))
            }));
        let workspace_count = workspace_slots.values().max().map_or(0, |&slot| slot + 1);
        let nodes: Vec<_> = order
            .iter()
            .map(|&id| {
                let node = &graph.nodes[id.index()];
                ScheduledNode {
                    id,
                    workspace: workspace_slots.get(&id).copied(),
                    input_slots: node.inputs.iter().map(|id| id.index()).collect(),
                    direct_unary: node
                        .operator
                        .as_ref()
                        .is_none_or(|op| op.supports_direct_unary()),
                }
            })
            .collect();
        let initial_uses = graph.nodes.iter().map(|node| node.last_use_count).collect();
        Ok(Self {
            root: graph.root()?.index(),
            entry_uses: nodes
                .iter()
                .filter(|node| node.input_slots.is_empty())
                .count(),
            nodes,
            initial_uses,
            workspace_count,
        })
    }
}

pub(crate) struct GraphScratch {
    remaining_uses: Vec<usize>,
    outputs: Vec<Option<Morsel>>,
    workspaces: Vec<Option<Box<dyn Any + Send + Sync>>>,
}

impl GraphScratch {
    fn new(slots: usize, workspace_count: usize) -> Self {
        Self {
            remaining_uses: vec![0; slots],
            outputs: vec![None; slots],
            workspaces: (0..workspace_count).map(|_| None).collect(),
        }
    }
}

/// Clear retained outputs on both a returned error and an unwind.
struct InvocationSlots<'a>(&'a mut [Option<Morsel>]);

impl Drop for InvocationSlots<'_> {
    fn drop(&mut self) {
        for slot in &mut *self.0 {
            *slot = None;
        }
    }
}

enum NodeInputs {
    Direct(Morsel),
    Ordered(Vec<Morsel>),
}

impl NodeInputs {
    fn as_slice(&self) -> &[Morsel] {
        match self {
            Self::Direct(input) => std::slice::from_ref(input),
            Self::Ordered(inputs) => inputs,
        }
    }
}

fn execute_graph(
    graph: &PhysicalGraph,
    schedule: &ExecutionSchedule,
    morsel: Morsel,
    profiler: &PhysicalProfiler,
    limits: crate::memory::GraphMemoryLimits,
    scratch: &mut GraphScratch,
    mut reservation: Option<&mut crate::memory::MemoryReservation>,
    cancelled: Option<&AtomicBool>,
) -> Result<Morsel, PhysicalGraphError> {
    let GraphScratch {
        remaining_uses,
        outputs,
        workspaces,
    } = scratch;
    remaining_uses.copy_from_slice(&schedule.initial_uses);
    let slots = InvocationSlots(outputs);
    let observe = profiler.is_enabled() || limits.max_observed_live_backing_bytes.is_some();
    let mut managed_backings = crate::memory::ManagedBackings::default();
    let mut tracked_managed_output = false;
    let mut entry_value = Some(morsel);
    let mut entry_uses = schedule.entry_uses;
    for scheduled in &schedule.nodes {
        if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(PhysicalGraphError::Cancelled);
        }
        let id = scheduled.id;
        let node = &graph.nodes[id.index()];
        let mut take_input = |slot_index| {
            remaining_uses[slot_index] -= 1;
            let slot: &mut Option<Morsel> = &mut slots.0[slot_index];
            let value = if remaining_uses[slot_index] == 0 {
                slot.take()
            } else {
                slot.clone()
            };
            value.ok_or_else(|| {
                PhysicalGraphError::UnsupportedExecutionShape(format!(
                    "input p{slot_index} has no execution value"
                ))
            })
        };
        let inputs = match scheduled.input_slots.as_slice() {
            [] => {
                entry_uses -= 1;
                let entry = if entry_uses == 0 {
                    entry_value.take()
                } else {
                    entry_value.clone()
                }
                .expect("scheduled graph entry");
                if scheduled.direct_unary {
                    NodeInputs::Direct(entry)
                } else {
                    NodeInputs::Ordered(vec![entry])
                }
            }
            [slot] if scheduled.direct_unary => NodeInputs::Direct(take_input(*slot)?),
            inputs => NodeInputs::Ordered(
                inputs
                    .iter()
                    .copied()
                    .map(take_input)
                    .collect::<Result<_, _>>()?,
            ),
        };
        if observe {
            observe_graph_memory(
                slots
                    .0
                    .iter()
                    .filter_map(Option::as_ref)
                    .chain(inputs.as_slice())
                    .chain(entry_value.iter()),
                profiler,
                limits,
                id,
            )?;
        }
        let managed_bound = if let (Some(reservation), Some(operator)) =
            (&reservation, &node.operator)
        {
            let bound = operator.managed_output_bytes(inputs.as_slice())?;
            if let Some(bytes) = bound {
                let remaining = reservation.bytes() - managed_backings.live_bytes();
                if bytes > remaining {
                    return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                        "p{} ({}) requires {bytes} managed output bytes before allocation; {remaining} bytes remain in this morsel's reservation",
                        id.index(),
                        operator.name(),
                    )));
                }
            }
            bound
        } else {
            None
        };
        let input_bytes = inputs
            .as_slice()
            .iter()
            .fold(0usize, |sum, input| sum.saturating_add(input.bytes));
        let started = profiler.is_enabled().then(Instant::now);
        let value = match (inputs, &node.operator) {
            (NodeInputs::Direct(input), Some(operator)) => match scheduled.workspace {
                Some(slot) => {
                    let workspace =
                        workspaces[slot].get_or_insert_with(|| operator.new_workspace());
                    operator.execute_in_workspace(input, workspace.as_mut())?
                }
                None => operator.execute(input)?,
            },
            (NodeInputs::Direct(input), None) => input,
            (NodeInputs::Ordered(inputs), Some(operator)) => operator.execute_inputs(inputs)?,
            (NodeInputs::Ordered(_), None) => {
                return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                    "multi-input node p{} requires an operator",
                    id.index(),
                )));
            }
        };
        if let (Some(bound), Some(operator)) = (managed_bound, &node.operator) {
            tracked_managed_output = true;
            let mut allocated = 0usize;
            operator.visit_managed_output(&value, &mut |tensor| {
                allocated = allocated.saturating_add(managed_backings.insert(tensor));
            });
            if allocated > bound {
                return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                    "p{} ({}) allocated {allocated} managed backing bytes above its declared bound of {bound}",
                    id.index(),
                    operator.name(),
                )));
            }
        }
        if let Some(started) = started {
            profiler.record(id, started.elapsed(), input_bytes, value.bytes);
        }
        if value.bytes > limits.max_value_logical_bytes {
            profiler.record_memory_failure(false);
            return Err(PhysicalGraphError::ValueLogicalLimit {
                node: id.index(),
                bytes: value.bytes,
                limit: limits.max_value_logical_bytes,
            });
        }
        if observe {
            observe_graph_memory(
                slots
                    .0
                    .iter()
                    .filter_map(Option::as_ref)
                    .chain(std::iter::once(&value))
                    .chain(entry_value.iter()),
                profiler,
                limits,
                id,
            )?;
        }
        slots.0[id.index()] = Some(value);
    }
    let output = slots.0[schedule.root]
        .take()
        .ok_or(PhysicalGraphError::MissingRoot)?;
    drop(slots);
    if tracked_managed_output && let Some(reservation) = &mut reservation {
        reservation.shrink_to(managed_backings.live_bytes());
    }
    Ok(output)
}

fn observe_graph_memory<'a>(
    morsels: impl IntoIterator<Item = &'a Morsel>,
    profiler: &PhysicalProfiler,
    limits: crate::memory::GraphMemoryLimits,
    node: PhysNodeId,
) -> Result<(), PhysicalGraphError> {
    let snapshot = crate::memory::observe_live_morsels(morsels);
    profiler.record_memory(snapshot);
    if let Some(limit) = limits.max_observed_live_backing_bytes {
        snapshot.check_observed_limit(limit).map_err(|source| {
            profiler.record_memory_failure(true);
            PhysicalGraphError::MemoryAccounting {
                node: node.index(),
                source,
            }
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_plan::{LogicalNode, LogicalPlan, NodeKind};

    struct TestLowering;

    impl PhysicalLowering for TestLowering {
        fn lower_node(
            &self,
            _logical_id: NodeId,
            node: &LogicalNode,
            _physical_inputs: &[PhysNodeId],
        ) -> RuntimeResult<PhysicalNodeSpec> {
            let kind = match node.kind() {
                NodeKind::Source => PhysicalNodeKind::Source,
                NodeKind::Index => PhysicalNodeKind::Sampler,
                NodeKind::Op => PhysicalNodeKind::Kernel(KernelStage::Sample),
                NodeKind::Batch => PhysicalNodeKind::Batch,
                NodeKind::Cache => PhysicalNodeKind::Cache,
                NodeKind::DeviceCut => {
                    return Err(RuntimeError::Message(
                        "DeviceCut requires device-aware physical lowering".to_owned(),
                    ));
                }
                NodeKind::Sink => PhysicalNodeKind::Sink,
            };
            Ok(PhysicalNodeSpec::new(kind, ExecutionLane::Cpu))
        }
    }

    struct AddBytes;

    impl PhysicalOperator for AddBytes {
        fn name(&self) -> &str {
            "AddBytes"
        }

        fn execute(&self, mut morsel: Morsel) -> RuntimeResult<Morsel> {
            morsel.bytes += 16;
            Ok(morsel)
        }
    }

    struct OrderedJoin;
    impl PhysicalOperator for OrderedJoin {
        fn name(&self) -> &str {
            "OrderedJoin"
        }
        fn execute(&self, _input: Morsel) -> RuntimeResult<Morsel> {
            panic!("join must use input ports")
        }
        fn execute_inputs(&self, inputs: Vec<Morsel>) -> RuntimeResult<Morsel> {
            assert_eq!(inputs.len(), 3);
            assert_eq!(inputs[0].bytes, 32);
            assert_eq!(inputs[1].bytes, 16);
            assert_eq!(inputs[2].bytes, 16);
            let mut output = inputs[0].clone();
            output.bytes = inputs.iter().map(|input| input.bytes).sum();
            Ok(output)
        }
    }

    #[test]
    fn diamond_lowering_and_execution_preserve_shared_inputs_and_ordered_ports() {
        let mut logical = LogicalPlan::new();
        let source = logical.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let shared = logical.add_node(LogicalNode::new(NodeKind::Op, [source], None));
        let left = logical.add_node(LogicalNode::new(NodeKind::Op, [shared], None));
        let join = logical.add_node(LogicalNode::new(
            NodeKind::Sink,
            [left, shared, shared],
            None,
        ));
        logical.set_root(join).unwrap();
        let mut graph = PhysicalGraph::lower(&logical, &TestLowering).unwrap();
        let ids = graph
            .nodes()
            .iter()
            .map(|node| (node.logical_id.unwrap(), node.id))
            .collect::<HashMap<_, _>>();
        graph.nodes[ids[&shared].index()].operator = Some(Arc::new(AddBytes));
        graph.nodes[ids[&left].index()].operator = Some(Arc::new(AddBytes));
        graph.nodes[ids[&join].index()].operator = Some(Arc::new(OrderedJoin));
        graph.validate().unwrap();
        assert_eq!(graph.node(ids[&shared]).unwrap().last_use_count, 3);
        let profiler = PhysicalProfiler::enabled(graph.nodes().len());
        let graph = ExecutableGraph::new(graph).unwrap();
        let mut scratch = graph.scratch();
        let result = graph
            .execute_reusing(
                Morsel::new(9, vec![4, 8]),
                &profiler,
                Default::default(),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(result.bytes, 64);
        assert_eq!(result.source_indices, [4, 8]);
        assert_eq!(profiler.snapshot()[&ids[&shared]].executions, 1);
        assert_eq!(profiler.snapshot()[&ids[&left]].executions, 1);
        assert_eq!(
            graph
                .execute_reusing(
                    Morsel::new(10, vec![1]),
                    &profiler,
                    Default::default(),
                    &mut scratch
                )
                .unwrap()
                .bytes,
            64
        );
    }

    #[test]
    fn reusable_slots_release_shared_values_after_errors_and_panics() {
        struct Fail;
        impl PhysicalOperator for Fail {
            fn name(&self) -> &str {
                "Fail"
            }
            fn execute(&self, input: Morsel) -> RuntimeResult<Morsel> {
                assert_ne!(input.sequence_id, 1, "test unwind");
                Err(RuntimeError::Message("test error".to_owned()))
            }
            fn supports_direct_unary(&self) -> bool {
                true
            }
        }
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
            [],
        );
        let fail = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            )
            .with_operator(Arc::new(Fail)),
            [source],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [fail, source],
        );
        graph.set_root(sink).unwrap();
        let prepared = ExecutableGraph::new(graph).unwrap();
        let mut scratch = prepared.scratch();
        for sequence in [0, 1] {
            let mut morsel = Morsel::new(sequence, vec![]);
            let backing = Arc::new(vec![42u8]);
            let weak = Arc::downgrade(&backing);
            morsel.values.push(ExecutionValue::Encoded(backing));
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prepared.execute_reusing(
                    morsel,
                    &PhysicalProfiler::default(),
                    Default::default(),
                    &mut scratch,
                )
            }));
            if sequence == 0 {
                assert!(outcome.unwrap().is_err());
            } else {
                assert!(outcome.is_err());
            }
            assert!(scratch.outputs.iter().all(Option::is_none));
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn multi_input_node_without_join_operator_fails_explicitly() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
            [],
        );
        let join = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [source, source],
        );
        graph.set_root(join).unwrap();
        assert!(
            GraphExecutor::execute(
                &mut graph,
                Morsel::new(0, vec![]),
                &PhysicalProfiler::default()
            )
            .unwrap_err()
            .to_string()
            .contains("requires an operator")
        );
    }

    #[test]
    fn logical_lowering_preserves_edges_and_computes_last_uses() {
        let mut logical = LogicalPlan::new();
        let source = logical.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let sample = logical.add_node(LogicalNode::new(NodeKind::Op, [source], None));
        let batch = logical.add_node(LogicalNode::new(NodeKind::Batch, [sample], None));
        let sink = logical.add_node(LogicalNode::new(NodeKind::Sink, [batch], None));
        logical.set_root(sink).unwrap();

        let mut physical = PhysicalGraph::lower(&logical, &TestLowering).unwrap();
        let explanation = physical.explain().unwrap();
        assert_eq!(
            explanation,
            concat!(
                "PhysicalGraph(root=p3)\n",
                "  p0 Source lane=Cpu parallelism=Serial logical=%0 <- [] last_uses=1 storage=SourceRead\n",
                "  p1 SampleKernel lane=Cpu parallelism=Serial logical=%1 <- [p0] last_uses=1 storage=Unspecified\n",
                "  p2 Batch lane=Cpu parallelism=Serial logical=%2 <- [p1] last_uses=1 storage=Materialize\n",
                "  p3 Sink lane=Cpu parallelism=Serial logical=%3 <- [p2] last_uses=1 storage=Unspecified\n",
            )
        );
        assert_eq!(physical.nodes()[0].last_use_count, 1);
    }

    #[test]
    fn graph_executor_profiles_operator_and_preserves_morsel_metadata() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io)
                .with_operator(Arc::new(AddBytes)),
            [],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [source],
        );
        graph.set_root(sink).unwrap();
        let profiler = PhysicalProfiler::enabled(graph.nodes().len());
        let morsel = Morsel::new(42, vec![7, 3]);

        let output = GraphExecutor::execute(&mut graph, morsel, &profiler).unwrap();
        assert_eq!(output.sequence_id, 42);
        assert_eq!(output.source_indices, [7, 3]);
        assert_eq!(output.bytes, 16);
        assert_eq!(profiler.snapshot()[&source].executions, 1);
        assert_eq!(profiler.snapshot()[&source].output_bytes, 16);
    }

    #[test]
    fn batch_kernels_are_placed_after_the_batch_barrier() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io),
            [],
        );
        let sample = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            ),
            [source],
        );
        let batch_kernel = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Batch),
                ExecutionLane::Cpu,
            )
            .with_batch_lift_equivalence(),
            [sample],
        );
        let batch = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            [batch_kernel],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [batch],
        );
        graph.set_root(sink).unwrap();

        graph.move_batch_kernels_after_batch().unwrap();

        assert_eq!(
            graph.topological_order().unwrap(),
            vec![source, sample, batch, batch_kernel, sink]
        );
    }

    #[test]
    fn batch_stage_alone_does_not_authorize_lifting() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
            [],
        );
        let kernel = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Batch),
                ExecutionLane::Cpu,
            ),
            [source],
        );
        let batch = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            [kernel],
        );
        graph.set_root(batch).unwrap();
        graph.move_batch_kernels_after_batch().unwrap();
        assert_eq!(graph.topological_order().unwrap(), [source, kernel, batch]);
    }

    #[test]
    fn lifting_never_crosses_an_intervening_sample_transform() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
            [],
        );
        let before = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Batch),
                ExecutionLane::Cpu,
            )
            .with_batch_lift_equivalence(),
            [source],
        );
        let sample = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            ),
            [before],
        );
        let after = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Batch),
                ExecutionLane::Cpu,
            )
            .with_batch_lift_equivalence(),
            [sample],
        );
        let batch = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            [after],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [batch],
        );
        graph.set_root(sink).unwrap();
        graph.move_batch_kernels_after_batch().unwrap();
        assert_eq!(
            graph.topological_order().unwrap(),
            [source, before, sample, batch, after, sink]
        );
    }

    #[test]
    fn lifting_preserves_shared_dependencies() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
            [],
        );
        let shared = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Batch),
                ExecutionLane::Cpu,
            )
            .with_batch_lift_equivalence(),
            [source],
        );
        let batch = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            [shared],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [batch, shared],
        );
        graph.set_root(sink).unwrap();
        let original = graph.topological_order().unwrap();
        graph.move_batch_kernels_after_batch().unwrap();
        assert_eq!(graph.topological_order().unwrap(), original);
        assert_eq!(graph.node(batch).unwrap().inputs, [shared]);
        assert_eq!(graph.node(sink).unwrap().inputs, [batch, shared]);
    }

    #[test]
    fn index_nodes_lower_to_sampler_before_source_reads() {
        let mut logical = LogicalPlan::new();
        let source = logical.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let index = logical.add_node(LogicalNode::new(NodeKind::Index, [source], None));
        let sample = logical.add_node(LogicalNode::new(NodeKind::Op, [index], None));
        let batch = logical.add_node(LogicalNode::new(NodeKind::Batch, [sample], None));
        let sink = logical.add_node(LogicalNode::new(NodeKind::Sink, [batch], None));
        logical.set_root(sink).unwrap();

        let graph = PhysicalGraph::lower(&logical, &TestLowering).unwrap();
        let order = graph.execution_order().unwrap();
        assert_eq!(
            graph.node(order[0]).unwrap().kind,
            PhysicalNodeKind::Sampler
        );
        assert_eq!(graph.node(order[1]).unwrap().kind, PhysicalNodeKind::Source);
    }

    #[test]
    fn lane_changes_insert_explicit_h2d_with_transfer_liveness() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io),
            [],
        );
        let batch = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            [source],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Sink,
                ExecutionLane::Device {
                    class: DeviceClass::Metal,
                    ordinal: 2,
                },
            ),
            [batch],
        );
        graph.set_root(sink).unwrap();

        let transfers = graph.insert_transfers_for_lane_changes().unwrap();
        assert_eq!(transfers.len(), 1);
        let transfer = graph.node(transfers[0]).unwrap();
        assert_eq!(
            transfer.kind,
            PhysicalNodeKind::Transfer(TransferKind::HostToDevice)
        );
        assert_eq!(transfer.lane, ExecutionLane::Transfer);
        assert_eq!(
            transfer.transfer_target,
            Some(ExecutionLane::Device {
                class: DeviceClass::Metal,
                ordinal: 2
            })
        );
        assert_eq!(transfer.inputs, [batch]);
        assert_eq!(graph.node(batch).unwrap().last_use_count, 1);
        graph.set_transfer_estimate(transfers[0], 512).unwrap();
        assert!(graph.explain().unwrap().contains(
            "Transfer(H2D) lane=Transfer parallelism=Serial target=Device { class: Metal, ordinal: 2 } estimated_transfer_bytes=512"
        ));
    }

    #[test]
    fn prepared_dag_rejects_missing_and_unbound_transfers() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io),
            [],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Sink,
                ExecutionLane::Device {
                    class: DeviceClass::Metal,
                    ordinal: 0,
                },
            ),
            [source],
        );
        graph.set_root(sink).unwrap();
        assert!(
            ExecutableGraph::new(graph.clone())
                .err()
                .unwrap()
                .to_string()
                .contains("needs an explicit bound transfer")
        );
        graph.insert_transfers_for_lane_changes().unwrap();
        assert!(
            ExecutableGraph::new(graph)
                .err()
                .unwrap()
                .to_string()
                .contains("has no bound operator")
        );
    }

    #[test]
    fn malformed_transfer_lane_and_target_is_rejected() {
        let mut graph = PhysicalGraph::new();
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
            [],
        );
        let transfer = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Transfer(TransferKind::HostToDevice),
                ExecutionLane::Transfer,
            )
            .with_transfer_target(ExecutionLane::Cpu),
            [source],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
            [transfer],
        );
        graph.set_root(sink).unwrap();
        assert!(graph.validate().is_err());
        assert_eq!(
            transfer_for_lanes(
                &ExecutionLane::Device {
                    class: DeviceClass::Metal,
                    ordinal: 0,
                },
                &ExecutionLane::Cpu,
            ),
            Some(TransferKind::DeviceToHost)
        );
        assert_eq!(
            transfer_for_lanes(
                &ExecutionLane::Device {
                    class: DeviceClass::Metal,
                    ordinal: 0,
                },
                &ExecutionLane::Device {
                    class: DeviceClass::Metal,
                    ordinal: 1,
                }
            ),
            Some(TransferKind::DeviceToDevice)
        );
    }
}
