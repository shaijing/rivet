//! Backend-independent logical planning for Rivet pipelines.
//!
//! This crate owns graph topology and planning decisions. Domain crates attach
//! semantic payloads through [`PlanPayload`]; payloads must not contain backend
//! runtime handles or executable tensor state.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use thiserror::Error;

mod optimizer;
mod placement;
mod properties;

pub use optimizer::{
    Diagnostic, FusionCandidate, FusionRule, OptimizationReport, OptimizerContext, OptimizerError,
    OptimizerPass, PassResult, PhysicalCandidate, PhysicalCandidateProvider, PlanPlugin,
    PlanRegistry, optimize, run_fixed_point,
};
pub use placement::{
    CostEstimate, KernelCapabilities, KernelCapability, KernelClass, KernelRequirements,
    MachineProfile, PlacementCandidate, PlacementError, PlacementPlan, place,
};

pub use properties::{
    AxisOrder, Contiguity, DataType, DeviceClass, InferenceError, Mutability, OperatorProperties,
    OperatorStage, PropertyAnnotations, PropertyInference, Representation, Residency, ShapeDim,
    ValueGranularity, ValueProperties, ValueShape,
};

/// Device selected by an explicit logical boundary. This contains only stable
/// planning data; runtime device/context/queue handles are resolved later.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceTarget {
    pub class: DeviceClass,
    pub ordinal: usize,
}

impl DeviceTarget {
    pub fn new(class: DeviceClass, ordinal: usize) -> Self {
        Self { class, ordinal }
    }

    pub fn cuda(ordinal: usize) -> Self {
        Self::new(DeviceClass::Cuda, ordinal)
    }

    pub fn metal(ordinal: usize) -> Self {
        Self::new(DeviceClass::Metal, ordinal)
    }
}

/// Fixed semantic boundary from the host region into an accelerator region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceCut {
    pub target: DeviceTarget,
}

impl DeviceCut {
    pub fn new(target: DeviceTarget) -> Self {
        Self { target }
    }
}

impl PlanPayload for DeviceCut {
    fn domain(&self) -> &'static str {
        "rivet"
    }

    fn name(&self) -> &'static str {
        match self.target.class {
            DeviceClass::Cuda => "DeviceCutToCuda",
            DeviceClass::Metal => "DeviceCutToMetal",
            _ => "DeviceCut",
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Type-erased semantic value carried by a logical node or plan context.
pub trait PlanPayload: Any + Send + Sync {
    fn domain(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
}

/// Stable arena index. IDs are local to the plan that created them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Error returned by checked logical-plan operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PlanError {
    #[error("logical plan has no root")]
    MissingRoot,
    #[error("node id {0} is outside the plan arena")]
    InvalidNodeId(usize),
    #[error("node {node} has invalid input node id {input}")]
    InvalidInput { node: usize, input: usize },
    #[error("logical plan contains a cycle through node {0}")]
    Cycle(usize),
    #[error("device cut node {0} must carry a rivet DeviceCut payload")]
    InvalidDeviceCutPayload(usize),
    #[error("device cut node {0} must have exactly one input")]
    InvalidDeviceCutArity(usize),
    #[error("device cut node {0} cannot target the CPU device class")]
    InvalidDeviceCutTarget(usize),
}

/// Stable logical node category. Domain details live in the erased payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Source,
    Index,
    Op,
    Batch,
    Cache,
    DeviceCut,
    Sink,
}

/// Small inline input list with capacity for two common edges and spillover
/// storage for future multi-input operators.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct NodeInputs {
    inline: [Option<NodeId>; 2],
    overflow: Vec<NodeId>,
    len: usize,
}

impl NodeInputs {
    pub fn new(inputs: impl IntoIterator<Item = NodeId>) -> Self {
        let mut out = Self::default();
        for input in inputs {
            if out.len < 2 {
                out.inline[out.len] = Some(input);
            } else {
                out.overflow.push(input);
            }
            out.len += 1;
        }
        out
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inline
            .iter()
            .take(self.len.min(2))
            .flatten()
            .copied()
            .chain(self.overflow.iter().copied())
    }

    pub fn get(&self, index: usize) -> Option<NodeId> {
        if index >= self.len {
            None
        } else if index < 2 {
            self.inline[index]
        } else {
            self.overflow.get(index - 2).copied()
        }
    }

    fn replace_input(&mut self, index: usize, input: NodeId) -> bool {
        if index >= self.len {
            return false;
        }
        if index < 2 {
            self.inline[index] = Some(input);
        } else {
            self.overflow[index - 2] = input;
        }
        true
    }
}

impl fmt::Debug for NodeInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A node in the logical graph.
#[derive(Clone)]
pub struct LogicalNode {
    kind: NodeKind,
    inputs: NodeInputs,
    payload: Option<Arc<dyn PlanPayload>>,
}

impl LogicalNode {
    pub fn new(
        kind: NodeKind,
        inputs: impl IntoIterator<Item = NodeId>,
        payload: Option<Arc<dyn PlanPayload>>,
    ) -> Self {
        Self {
            kind,
            inputs: NodeInputs::new(inputs),
            payload,
        }
    }

    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    pub fn inputs(&self) -> &NodeInputs {
        &self.inputs
    }

    pub fn payload(&self) -> Option<&Arc<dyn PlanPayload>> {
        self.payload.as_ref()
    }

    pub fn payload_as<T: Any>(&self) -> Option<&T> {
        self.payload.as_ref()?.as_any().downcast_ref()
    }
}

impl fmt::Debug for LogicalNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("LogicalNode");
        debug
            .field("kind", &self.kind)
            .field("inputs", &self.inputs);
        if let Some(payload) = &self.payload {
            debug.field(
                "payload",
                &format_args!("{}::{}", payload.domain(), payload.name()),
            );
        }
        debug.finish()
    }
}

/// Append-only arena for logical nodes; replacement preserves node IDs.
#[derive(Clone, Default)]
pub struct PlanArena {
    nodes: Vec<LogicalNode>,
}

impl PlanArena {
    pub fn add(&mut self, node: LogicalNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn replace(&mut self, id: NodeId, node: LogicalNode) -> Result<(), PlanError> {
        let slot = self
            .nodes
            .get_mut(id.0)
            .ok_or(PlanError::InvalidNodeId(id.0))?;
        *slot = node;
        Ok(())
    }

    pub fn get(&self, id: NodeId) -> Result<&LogicalNode, PlanError> {
        self.nodes.get(id.0).ok_or(PlanError::InvalidNodeId(id.0))
    }

    pub fn get_mut(&mut self, id: NodeId) -> Result<&mut LogicalNode, PlanError> {
        self.nodes
            .get_mut(id.0)
            .ok_or(PlanError::InvalidNodeId(id.0))
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (NodeId, &LogicalNode)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (NodeId(i), node))
    }
}

/// Owned arena-backed logical plan.
#[derive(Clone, Default)]
pub struct LogicalPlan {
    root: Option<NodeId>,
    arena: PlanArena,
    context: Option<Arc<dyn PlanPayload>>,
}

impl LogicalPlan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: LogicalNode) -> NodeId {
        self.arena.add(node)
    }

    pub fn replace_node(&mut self, id: NodeId, node: LogicalNode) -> Result<(), PlanError> {
        self.arena.replace(id, node)
    }

    /// Replace all edges to `old` with `new` and update the root if necessary.
    pub fn redirect_uses(&mut self, old: NodeId, new: NodeId) -> Result<(), PlanError> {
        self.arena.get(old)?;
        self.arena.get(new)?;
        for node in &mut self.arena.nodes {
            let positions = node
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(index, input)| (input == old).then_some(index))
                .collect::<Vec<_>>();
            for index in positions {
                node.inputs.replace_input(index, new);
            }
        }
        if self.root == Some(old) {
            self.root = Some(new);
        }
        Ok(())
    }

    /// Compact the arena to root-reachable nodes and return the old-to-new ID map.
    /// Callers must discard node annotations after compaction because IDs change.
    pub fn prune_unreachable(&mut self) -> Result<Vec<Option<NodeId>>, PlanError> {
        let preorder = self.preorder()?;
        let mut is_reachable = vec![false; self.arena.len()];
        for id in preorder {
            is_reachable[id.index()] = true;
        }
        // Keep existing arena order for surviving nodes. This makes ordinary
        // pipelines retain stable IDs while removing dead branches.
        let reachable = (0..self.arena.len())
            .filter(|index| is_reachable[*index])
            .map(NodeId)
            .collect::<Vec<_>>();
        let mut remap = vec![None; self.arena.len()];
        for (new, old) in reachable.iter().enumerate() {
            remap[old.index()] = Some(NodeId(new));
        }
        let old_nodes = std::mem::take(&mut self.arena.nodes);
        self.arena.nodes = reachable
            .iter()
            .map(|old| {
                let node = &old_nodes[old.index()];
                let inputs = node
                    .inputs
                    .iter()
                    .map(|id| remap[id.index()].expect("reachable input"));
                LogicalNode::new(node.kind, inputs, node.payload.clone())
            })
            .collect();
        self.root = self
            .root
            .map(|id| remap[id.index()].expect("root is reachable"));
        Ok(remap)
    }

    /// Validate and canonicalize the arena by dropping unreachable nodes while
    /// preserving insertion order and therefore stable IDs for live nodes.
    pub fn canonicalize(&mut self) -> Result<Vec<Option<NodeId>>, PlanError> {
        self.validate()?;
        self.prune_unreachable()
    }

    pub fn node(&self, id: NodeId) -> Result<&LogicalNode, PlanError> {
        self.arena.get(id)
    }

    pub fn nodes(&self) -> impl ExactSizeIterator<Item = (NodeId, &LogicalNode)> {
        self.arena.iter()
    }

    pub fn set_root(&mut self, root: NodeId) -> Result<(), PlanError> {
        self.arena.get(root)?;
        self.root = Some(root);
        Ok(())
    }

    pub fn root(&self) -> Result<NodeId, PlanError> {
        self.root.ok_or(PlanError::MissingRoot)
    }

    pub fn set_context(&mut self, context: Arc<dyn PlanPayload>) {
        self.context = Some(context);
    }

    pub fn context_as<T: Any>(&self) -> Option<&T> {
        self.context.as_ref()?.as_any().downcast_ref()
    }

    /// Direct input dependencies of this node, in declared input order.
    pub fn parents(&self, id: NodeId) -> Result<Vec<NodeId>, PlanError> {
        Ok(self.arena.get(id)?.inputs.iter().collect())
    }

    /// Direct consumers of this node, in arena insertion order.
    pub fn children(&self, id: NodeId) -> Result<Vec<NodeId>, PlanError> {
        self.arena.get(id)?;
        Ok(self
            .arena
            .iter()
            .filter_map(|(child, node)| {
                node.inputs.iter().any(|input| input == id).then_some(child)
            })
            .collect())
    }

    /// Deterministic depth-first preorder, following ordered input edges.
    pub fn preorder(&self) -> Result<Vec<NodeId>, PlanError> {
        let root = self.root()?;
        let mut stack = vec![root];
        let mut seen = vec![false; self.arena.len()];
        let mut out = Vec::new();
        while let Some(id) = stack.pop() {
            let node = self.arena.get(id)?;
            if seen[id.0] {
                continue;
            }
            seen[id.0] = true;
            out.push(id);
            for child in node.inputs.iter().collect::<Vec<_>>().into_iter().rev() {
                stack.push(child);
            }
        }
        Ok(out)
    }

    pub fn validate(&self) -> Result<(), PlanError> {
        let root = self.root()?;
        self.arena.get(root)?;
        for (node_id, node) in self.arena.iter() {
            if node.kind == NodeKind::DeviceCut {
                if node.inputs.len() != 1 {
                    return Err(PlanError::InvalidDeviceCutArity(node_id.index()));
                }
                let Some(cut) = node.payload_as::<DeviceCut>() else {
                    return Err(PlanError::InvalidDeviceCutPayload(node_id.index()));
                };
                if cut.target.class == DeviceClass::Cpu {
                    return Err(PlanError::InvalidDeviceCutTarget(node_id.index()));
                }
            }
            for input in node.inputs.iter() {
                if input.0 >= self.arena.len() {
                    return Err(PlanError::InvalidInput {
                        node: node_id.0,
                        input: input.0,
                    });
                }
            }
        }
        let mut state = vec![0u8; self.arena.len()];
        fn visit(id: NodeId, arena: &PlanArena, state: &mut [u8]) -> Result<(), PlanError> {
            match state[id.0] {
                1 => return Err(PlanError::Cycle(id.0)),
                2 => return Ok(()),
                _ => {}
            }
            state[id.0] = 1;
            for input in arena.get(id)?.inputs.iter() {
                visit(input, arena, state)?;
            }
            state[id.0] = 2;
            Ok(())
        }
        visit(root, &self.arena, &mut state)?;
        for (id, _) in self.arena.iter() {
            visit(id, &self.arena, &mut state)?;
        }
        Ok(())
    }

    pub fn explain(&self) -> Result<String, PlanError> {
        self.validate()?;
        let root = self.root()?;
        let mut out = format!("LogicalPlan(root=%{})\n", root.0);
        for (id, node) in self.arena.iter() {
            let label = if let Some(cut) = node.payload_as::<DeviceCut>() {
                match &cut.target.class {
                    DeviceClass::Cuda | DeviceClass::Metal => {
                        format!(" rivet::{}({})", cut.name(), cut.target.ordinal)
                    }
                    class => format!(
                        " rivet::DeviceCut({class:?}, ordinal={})",
                        cut.target.ordinal
                    ),
                }
            } else {
                node.payload
                    .as_ref()
                    .map(|p| format!(" {}::{}", p.domain(), p.name()))
                    .unwrap_or_default()
            };
            let inputs = node
                .inputs
                .iter()
                .map(|id| format!("%{}", id.0))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "  %{} {:?}{} <- [{}]\n",
                id.0, node.kind, label, inputs
            ));
        }
        Ok(out)
    }

    /// Infer and validate value properties in input-before-consumer order.
    pub fn infer_properties(
        &self,
        inference: &dyn PropertyInference,
    ) -> Result<PropertyAnnotations, InferenceError> {
        self.validate().map_err(InferenceError::InvalidPlan)?;
        let mut annotations = PropertyAnnotations::default();
        let mut active = vec![false; self.arena.len()];
        fn infer_node(
            plan: &LogicalPlan,
            id: NodeId,
            inference: &dyn PropertyInference,
            annotations: &mut PropertyAnnotations,
            active: &mut [bool],
        ) -> Result<(), InferenceError> {
            if annotations.values.contains_key(&id) {
                return Ok(());
            }
            if active[id.index()] {
                return Err(InferenceError::Cycle(id));
            }
            active[id.index()] = true;
            let node = plan.node(id).map_err(InferenceError::InvalidPlan)?;
            let mut inputs = Vec::with_capacity(node.inputs().len());
            for input in node.inputs().iter() {
                infer_node(plan, input, inference, annotations, active)?;
                inputs.push(annotations.values[&input].clone());
            }
            let properties = inference
                .infer_node(node, &inputs)
                .map_err(|message| InferenceError::Node { node: id, message })?;
            annotations.values.insert(id, properties);
            active[id.index()] = false;
            Ok(())
        }
        infer_node(
            self,
            self.root().map_err(InferenceError::InvalidPlan)?,
            inference,
            &mut annotations,
            &mut active,
        )?;
        Ok(annotations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPayload;
    impl PlanPayload for TestPayload {
        fn domain(&self) -> &'static str {
            "test"
        }
        fn name(&self) -> &'static str {
            "op"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn explain_and_traversal_are_stable() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let op = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(TestPayload)),
        ));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [op], None));
        plan.set_root(sink).unwrap();
        assert_eq!(plan.preorder().unwrap(), [sink, op, source]);
        assert_eq!(plan.parents(op).unwrap(), [source]);
        assert_eq!(plan.children(op).unwrap(), [sink]);
        assert_eq!(
            plan.explain().unwrap(),
            "LogicalPlan(root=%2)\n  %0 Source <- []\n  %1 Op test::op <- [%0]\n  %2 Sink <- [%1]\n"
        );
    }

    #[test]
    fn device_cut_is_plain_target_data_and_participates_in_plan_traversal() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let cuda_cut = plan.add_node(LogicalNode::new(
            NodeKind::DeviceCut,
            [source],
            Some(Arc::new(DeviceCut::new(DeviceTarget::cuda(3)))),
        ));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [cuda_cut], None));
        plan.set_root(sink).unwrap();

        let traversal = plan.preorder().unwrap();
        assert_eq!(traversal, [sink, cuda_cut, source]);
        assert_eq!(plan.clone().preorder().unwrap(), traversal);
        assert_eq!(plan.parents(cuda_cut).unwrap(), [source]);
        assert_eq!(plan.children(cuda_cut).unwrap(), [sink]);
        let explanation = plan.explain().unwrap();
        assert!(explanation.contains("DeviceCutToCuda(3)"), "{explanation}");

        let cuda = plan
            .node(cuda_cut)
            .unwrap()
            .payload_as::<DeviceCut>()
            .unwrap();
        assert_eq!(cuda.target, DeviceTarget::cuda(3));
        assert_eq!(cuda.target.class, DeviceClass::Cuda);
        assert_eq!(cuda.target.ordinal, 3);

        let metal = DeviceCut::new(DeviceTarget::metal(1));
        assert_eq!(metal.name(), "DeviceCutToMetal");
        assert_eq!(metal.target.class, DeviceClass::Metal);
    }

    #[test]
    fn device_cut_validation_rejects_missing_payload_bad_arity_and_cpu_target() {
        let mut missing_payload = LogicalPlan::new();
        let source = missing_payload.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let cut = missing_payload.add_node(LogicalNode::new(NodeKind::DeviceCut, [source], None));
        let sink = missing_payload.add_node(LogicalNode::new(NodeKind::Sink, [cut], None));
        missing_payload.set_root(sink).unwrap();
        assert_eq!(
            missing_payload.validate(),
            Err(PlanError::InvalidDeviceCutPayload(cut.index()))
        );

        let mut bad_arity = LogicalPlan::new();
        let source = bad_arity.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let cut = bad_arity.add_node(LogicalNode::new(
            NodeKind::DeviceCut,
            [source, source],
            Some(Arc::new(DeviceCut::new(DeviceTarget::cuda(0)))),
        ));
        let sink = bad_arity.add_node(LogicalNode::new(NodeKind::Sink, [cut], None));
        bad_arity.set_root(sink).unwrap();
        assert_eq!(
            bad_arity.validate(),
            Err(PlanError::InvalidDeviceCutArity(cut.index()))
        );

        let mut cpu_target = LogicalPlan::new();
        let source = cpu_target.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let cut = cpu_target.add_node(LogicalNode::new(
            NodeKind::DeviceCut,
            [source],
            Some(Arc::new(DeviceCut::new(DeviceTarget::new(
                DeviceClass::Cpu,
                0,
            )))),
        ));
        let sink = cpu_target.add_node(LogicalNode::new(NodeKind::Sink, [cut], None));
        cpu_target.set_root(sink).unwrap();
        assert_eq!(
            cpu_target.validate(),
            Err(PlanError::InvalidDeviceCutTarget(cut.index()))
        );
    }

    #[test]
    fn replacement_keeps_id_and_validation_rejects_cycle() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(source).unwrap();
        plan.replace_node(source, LogicalNode::new(NodeKind::Op, [source], None))
            .unwrap();
        assert_eq!(plan.validate(), Err(PlanError::Cycle(source.index())));
    }

    #[test]
    fn canonicalization_drops_unreachable_nodes_and_keeps_live_arena_order() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let _dead = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let op = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(TestPayload)),
        ));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [op], None));
        plan.set_root(sink).unwrap();

        let mapping = plan.canonicalize().unwrap();
        assert_eq!(
            mapping,
            [Some(NodeId(0)), None, Some(NodeId(1)), Some(NodeId(2))]
        );
        assert_eq!(plan.root().unwrap(), NodeId(2));
        assert_eq!(plan.preorder().unwrap(), [NodeId(2), NodeId(1), NodeId(0)]);
        assert_eq!(
            plan.node(NodeId(1)).unwrap().inputs().get(0),
            Some(NodeId(0))
        );
        assert_eq!(plan.arena.len(), 3);
        assert!(plan.validate().is_ok());
    }
}
