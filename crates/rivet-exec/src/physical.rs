//! Domain-neutral physical graph and morsel execution protocol.
//!
//! Logical plans describe semantic operations. This module lowers their
//! topology to executable physical nodes without putting backend handles in
//! `rivet-plan`.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
#[derive(Clone)]
pub enum ExecutionValue {
    Encoded(Arc<dyn Any + Send + Sync>),
    Decoded(Arc<dyn Any + Send + Sync>),
    Tensor(Tensor),
}

impl fmt::Debug for ExecutionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoded(_) => f.write_str("Encoded(..)"),
            Self::Decoded(_) => f.write_str("Decoded(..)"),
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

/// Output for one logical node during lowering. A lowerer may fuse multiple
/// logical nodes in a later extension; the initial contract maps one logical
/// node to one physical node.
pub struct PhysicalNodeSpec {
    pub kind: PhysicalNodeKind,
    pub lane: ExecutionLane,
    pub operator: Option<Arc<dyn PhysicalOperator>>,
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
    pub transfer_target: Option<ExecutionLane>,
    pub estimated_transfer_bytes: Option<u64>,
    pub parallelism: ParallelismClass,
    pub memory_requirements: KernelMemoryRequirements,
    pub estimated_output_bytes: Option<usize>,
    /// Domain proof that applying this unary kernel per sample is equivalent
    /// to applying it after stacking. Stage classification alone is insufficient.
    pub batch_lift_equivalent: bool,
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
}

impl PhysicalGraph {
    pub fn new() -> Self {
        Self::default()
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
        let root = self.root()?;
        let mut out = format!("PhysicalGraph(root=p{})\n", root.index());
        for id in self.topological_order()? {
            let node = self.node(id)?;
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
                .map(|op| format!(" op={}", op.name()))
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
            out.push_str(&format!(
                "  p{} {} lane={:?} parallelism={:?}{}{}{}{} <- [{}] last_uses={}\n",
                node.id.index(),
                node.kind,
                node.lane,
                node.parallelism,
                logical,
                operator,
                transfer_target,
                transfer_estimate,
                inputs,
                node.last_use_count
            ));
        }
        Ok(out)
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
    pub input_bytes: u64,
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

#[derive(Clone)]
pub struct PhysicalProfiler {
    nodes: Option<Arc<[AtomicNodeProfile]>>,
}

impl PhysicalProfiler {
    /// Create an enabled profiler with one dense counter slot per physical node.
    /// Profiling is opt-in because even atomic accounting perturbs short kernels.
    pub fn enabled(node_count: usize) -> Self {
        let nodes = (0..node_count)
            .map(|_| AtomicNodeProfile::default())
            .collect::<Vec<_>>()
            .into();
        Self { nodes: Some(nodes) }
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
        Self { nodes: None }
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
}

impl ExecutableGraph {
    pub fn new(mut graph: PhysicalGraph) -> Result<Self, PhysicalGraphError> {
        graph.validate()?;
        let order = graph.execution_order()?;
        Ok(Self { graph, order })
    }

    pub fn execute_with_limit(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
        max_bytes: usize,
    ) -> Result<Morsel, PhysicalGraphError> {
        execute_graph(&self.graph, &self.order, morsel, profiler, max_bytes)
    }

    pub fn graph(&self) -> &PhysicalGraph {
        &self.graph
    }

    pub fn execute(
        &self,
        morsel: Morsel,
        profiler: &PhysicalProfiler,
    ) -> Result<Morsel, PhysicalGraphError> {
        execute_graph(&self.graph, &self.order, morsel, profiler, usize::MAX)
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
        execute_graph(
            graph,
            &graph.execution_order()?,
            morsel,
            profiler,
            usize::MAX,
        )
    }
}

fn execute_graph(
    graph: &PhysicalGraph,
    order: &[PhysNodeId],
    morsel: Morsel,
    profiler: &PhysicalProfiler,
    max_bytes: usize,
) -> Result<Morsel, PhysicalGraphError> {
    let mut remaining_uses = graph
        .nodes
        .iter()
        .map(|node| node.last_use_count)
        .collect::<Vec<_>>();
    let root = graph.root()?;
    let mut outputs = vec![None::<Morsel>; graph.nodes.len()];
    for &id in order {
        let node = graph.node(id)?;
        let mut inputs = Vec::with_capacity(node.inputs.len().max(1));
        for &input in &node.inputs {
            remaining_uses[input.index()] -= 1;
            let slot = &mut outputs[input.index()];
            let value = if remaining_uses[input.index()] == 0 {
                slot.take()
            } else {
                slot.clone()
            };
            inputs.push(value.ok_or_else(|| {
                PhysicalGraphError::UnsupportedExecutionShape(format!(
                    "input p{} has no execution value",
                    input.index()
                ))
            })?);
        }
        if inputs.is_empty() {
            inputs.push(morsel.clone());
        }
        let input_bytes = inputs.iter().map(|input| input.bytes).sum();
        let started = profiler.is_enabled().then(Instant::now);
        let value = if let Some(operator) = &node.operator {
            operator
                .execute_inputs(inputs)
                .map_err(PhysicalGraphError::Operator)?
        } else if inputs.len() == 1 {
            inputs.pop().expect("checked unary input")
        } else {
            return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                "multi-input node p{} requires an operator",
                id.index()
            )));
        };
        if let Some(started) = started {
            profiler.record(id, started.elapsed(), input_bytes, value.bytes);
        }
        if value.bytes > max_bytes {
            return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                "node p{} output exceeds DAG edge byte limit: {} > {}",
                id.index(),
                value.bytes,
                max_bytes
            )));
        }
        outputs[id.index()] = Some(value);
    }
    outputs[root.index()]
        .take()
        .ok_or(PhysicalGraphError::MissingRoot)
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
        let result = graph
            .execute(Morsel::new(9, vec![4, 8]), &profiler)
            .unwrap();
        assert_eq!(result.bytes, 64);
        assert_eq!(result.source_indices, [4, 8]);
        assert_eq!(profiler.snapshot()[&ids[&shared]].executions, 1);
        assert_eq!(profiler.snapshot()[&ids[&left]].executions, 1);
        assert_eq!(
            graph
                .execute(Morsel::new(10, vec![1]), &profiler)
                .unwrap()
                .bytes,
            64
        );
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
                "  p0 Source lane=Cpu parallelism=Serial logical=%0 <- [] last_uses=1\n",
                "  p1 SampleKernel lane=Cpu parallelism=Serial logical=%1 <- [p0] last_uses=1\n",
                "  p2 Batch lane=Cpu parallelism=Serial logical=%2 <- [p1] last_uses=1\n",
                "  p3 Sink lane=Cpu parallelism=Serial logical=%3 <- [p2] last_uses=1\n",
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
                    class: DeviceClass::Cuda,
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
                class: DeviceClass::Cuda,
                ordinal: 2
            })
        );
        assert_eq!(transfer.inputs, [batch]);
        assert_eq!(graph.node(batch).unwrap().last_use_count, 1);
        graph.set_transfer_estimate(transfers[0], 512).unwrap();
        assert!(graph.explain().unwrap().contains(
            "Transfer(H2D) lane=Transfer parallelism=Serial target=Device { class: Cuda, ordinal: 2 } estimated_transfer_bytes=512"
        ));
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
                    class: DeviceClass::Cuda,
                    ordinal: 0,
                },
                &ExecutionLane::Cpu,
            ),
            Some(TransferKind::DeviceToHost)
        );
        assert_eq!(
            transfer_for_lanes(
                &ExecutionLane::Device {
                    class: DeviceClass::Cuda,
                    ordinal: 0,
                },
                &ExecutionLane::Device {
                    class: DeviceClass::Cuda,
                    ordinal: 1,
                }
            ),
            Some(TransferKind::DeviceToDevice)
        );
    }
}
