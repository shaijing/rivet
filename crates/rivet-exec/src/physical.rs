//! Domain-neutral physical graph and morsel execution protocol.
//!
//! Logical plans describe semantic operations. This module lowers their
//! topology to executable physical nodes without putting backend handles in
//! `rivet-plan`.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rivet_core::{DType, Tensor};
use rivet_plan::{LogicalNode, LogicalPlan, NodeId, PlanError, ValueGranularity};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionLane {
    Io,
    Cpu,
    Transfer,
    Device { ordinal: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelStage {
    Decode,
    Sample,
    Batch,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorselResidency {
    Host,
    Device { ordinal: usize },
    Unknown,
}

/// Domain kernel or IO operation attached to a physical node.
pub trait PhysicalOperator: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, morsel: Morsel) -> RuntimeResult<Morsel>;
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
}

impl PhysicalNodeSpec {
    pub fn new(kind: PhysicalNodeKind, lane: ExecutionLane) -> Self {
        Self {
            kind,
            lane,
            operator: None,
            transfer_target: None,
            estimated_transfer_bytes: None,
        }
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
    #[error("physical graph execution currently requires a single linear path: {0}")]
    UnsupportedExecutionShape(String),
    #[error("physical operator failed: {0}")]
    Operator(#[from] RuntimeError),
}

/// Owned physical graph. `last_use_count` is computed when validating or
/// lowering the graph and can drive buffer release decisions in later stages.
#[derive(Default)]
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

    pub fn execution_order(&self) -> Result<Vec<PhysNodeId>, PhysicalGraphError> {
        self.topological_order()
    }

    /// Add explicit transfer nodes for host/device lane changes on the
    /// initial linear physical path. Host I/O and CPU lanes share residency;
    /// a future multi-input runtime can extend this edge-local lowering.
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
            let consumer_lane = self.node(consumer)?.lane;
            let inputs = self.node(consumer)?.inputs.clone();
            for (input_position, input) in inputs.into_iter().enumerate() {
                let producer_lane = output_lane(self.node(input)?);
                let Some(kind) = transfer_for_lanes(producer_lane, consumer_lane) else {
                    continue;
                };
                let transfer = self.add_node(
                    None,
                    PhysicalNodeSpec::new(
                        PhysicalNodeKind::Transfer(kind),
                        ExecutionLane::Transfer,
                    )
                    .with_transfer_target(consumer_lane),
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
        // LogicalPlan::preorder is consumer-first from its root. Reversing it
        // gives input-before-consumer order, including shared dependencies.
        for logical_id in logical.preorder()?.into_iter().rev() {
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
                        source_lane,
                        node.transfer_target.expect("checked transfer target"),
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
                    let target = producer.transfer_target.expect("validated transfer target");
                    if !lanes_match(target, node.lane) {
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
                .map(|target| format!(" target={target:?}"))
                .unwrap_or_default();
            let transfer_estimate = node
                .estimated_transfer_bytes
                .map(|bytes| format!(" estimated_transfer_bytes={bytes}"))
                .unwrap_or_default();
            out.push_str(&format!(
                "  p{} {} lane={:?}{}{}{}{} <- [{}] last_uses={}\n",
                node.id.index(),
                node.kind,
                node.lane,
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

    /// Logical image pipelines currently keep semantic ops in their declared
    /// order and annotate batch-stage ops before the explicit batch barrier.
    /// Convert that compatibility form into the physical execution order for
    /// a single linear pipeline.
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
        let moved = order[..batch_index]
            .iter()
            .copied()
            .filter(|id| self.nodes[id.0].kind == PhysicalNodeKind::Kernel(KernelStage::Batch))
            .collect::<Vec<_>>();
        if moved.is_empty() {
            return Ok(());
        }

        let moved_ids = moved
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut reordered = order[..batch_index]
            .iter()
            .copied()
            .filter(|id| !moved_ids.contains(id))
            .collect::<Vec<_>>();
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
        let Some(source_position) = order
            .iter()
            .position(|id| self.nodes[id.0].kind == PhysicalNodeKind::Source)
        else {
            return Ok(());
        };
        let samplers = order
            .iter()
            .copied()
            .filter(|id| self.nodes[id.0].kind == PhysicalNodeKind::Sampler)
            .collect::<Vec<_>>();
        if samplers.is_empty()
            || samplers.iter().all(|id| {
                order
                    .iter()
                    .position(|ordered| ordered == id)
                    .is_some_and(|position| position < source_position)
            })
        {
            return Ok(());
        }

        let sampler_ids = samplers
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut reordered = samplers;
        reordered.extend(order.iter().copied().filter(|id| !sampler_ids.contains(id)));
        for (position, id) in reordered.iter().copied().enumerate() {
            self.nodes[id.0].inputs = if position == 0 {
                Vec::new()
            } else {
                vec![reordered[position - 1]]
            };
        }
        Ok(())
    }
}

fn transfer_for_lanes(from: ExecutionLane, to: ExecutionLane) -> Option<TransferKind> {
    match (from, to) {
        (ExecutionLane::Cpu | ExecutionLane::Io, ExecutionLane::Device { .. }) => {
            Some(TransferKind::HostToDevice)
        }
        (ExecutionLane::Device { .. }, ExecutionLane::Cpu | ExecutionLane::Io) => {
            Some(TransferKind::DeviceToHost)
        }
        (ExecutionLane::Device { ordinal: from }, ExecutionLane::Device { ordinal: to })
            if from != to =>
        {
            Some(TransferKind::DeviceToDevice)
        }
        _ => None,
    }
}

fn output_lane(node: &PhysicalNode) -> ExecutionLane {
    node.transfer_target.unwrap_or(node.lane)
}

fn lanes_match(lhs: ExecutionLane, rhs: ExecutionLane) -> bool {
    match (lhs, rhs) {
        (ExecutionLane::Cpu | ExecutionLane::Io, ExecutionLane::Cpu | ExecutionLane::Io) => true,
        (ExecutionLane::Device { ordinal: lhs }, ExecutionLane::Device { ordinal: rhs }) => {
            lhs == rhs
        }
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

#[derive(Clone, Default)]
pub struct PhysicalProfiler {
    nodes: Arc<Mutex<HashMap<PhysNodeId, NodeProfile>>>,
}

impl PhysicalProfiler {
    pub fn snapshot(&self) -> HashMap<PhysNodeId, NodeProfile> {
        self.nodes
            .lock()
            .expect("physical profiler mutex poisoned")
            .clone()
    }

    pub(crate) fn record(&self, id: PhysNodeId, elapsed: Duration, input: usize, output: usize) {
        let mut nodes = self.nodes.lock().expect("physical profiler mutex poisoned");
        let profile = nodes.entry(id).or_default();
        profile.executions += 1;
        profile.elapsed += elapsed;
        profile.input_bytes += input as u64;
        profile.output_bytes += output as u64;
    }

    pub(crate) fn record_wait(&self, id: PhysNodeId, elapsed: Duration) {
        if elapsed.is_zero() {
            return;
        }
        let mut nodes = self.nodes.lock().expect("physical profiler mutex poisoned");
        let profile = nodes.entry(id).or_default();
        profile.wait_elapsed += elapsed;
        profile.wait_events += 1;
    }
}

/// Executes attached operators along the initial single-path physical graph.
/// The graph representation already tracks DAG edges and last uses; fan-out
/// execution will be enabled once value ownership policies are specified.
pub struct GraphExecutor;

impl GraphExecutor {
    pub fn execute(
        graph: &mut PhysicalGraph,
        mut morsel: Morsel,
        profiler: &PhysicalProfiler,
    ) -> Result<Morsel, PhysicalGraphError> {
        graph.validate()?;
        let order = graph.topological_order()?;
        let mut consumers = vec![0usize; graph.nodes.len()];
        for node in &graph.nodes {
            if node.inputs.len() > 1 {
                return Err(PhysicalGraphError::UnsupportedExecutionShape(format!(
                    "p{} has {} inputs",
                    node.id.index(),
                    node.inputs.len()
                )));
            }
            for input in &node.inputs {
                consumers[input.0] += 1;
            }
        }
        if consumers.iter().any(|count| *count > 1) {
            return Err(PhysicalGraphError::UnsupportedExecutionShape(
                "fan-out requires a value-specific clone policy".to_string(),
            ));
        }
        for id in order {
            let node = graph.node(id)?;
            let Some(operator) = &node.operator else {
                continue;
            };
            let input_bytes = morsel.bytes;
            let started = Instant::now();
            morsel = operator
                .execute(morsel)
                .map_err(PhysicalGraphError::Operator)?;
            profiler.record(id, started.elapsed(), input_bytes, morsel.bytes);
        }
        Ok(morsel)
    }
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
                "  p0 Source lane=Cpu logical=%0 <- [] last_uses=1\n",
                "  p1 SampleKernel lane=Cpu logical=%1 <- [p0] last_uses=1\n",
                "  p2 Batch lane=Cpu logical=%2 <- [p1] last_uses=1\n",
                "  p3 Sink lane=Cpu logical=%3 <- [p2] last_uses=1\n",
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
        let profiler = PhysicalProfiler::default();
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
            ),
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
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Device { ordinal: 2 }),
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
            Some(ExecutionLane::Device { ordinal: 2 })
        );
        assert_eq!(transfer.inputs, [batch]);
        assert_eq!(graph.node(batch).unwrap().last_use_count, 1);
        graph.set_transfer_estimate(transfers[0], 512).unwrap();
        assert!(graph.explain().unwrap().contains(
            "Transfer(H2D) lane=Transfer target=Device { ordinal: 2 } estimated_transfer_bytes=512"
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
            transfer_for_lanes(ExecutionLane::Device { ordinal: 0 }, ExecutionLane::Cpu),
            Some(TransferKind::DeviceToHost)
        );
        assert_eq!(
            transfer_for_lanes(
                ExecutionLane::Device { ordinal: 0 },
                ExecutionLane::Device { ordinal: 1 }
            ),
            Some(TransferKind::DeviceToDevice)
        );
    }
}
