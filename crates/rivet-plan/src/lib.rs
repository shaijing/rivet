//! Backend-independent logical planning for Rivet pipelines.
//!
//! This crate owns graph topology and planning decisions. Domain crates attach
//! semantic payloads through [`PlanPayload`]; payloads must not contain backend
//! runtime handles or executable tensor state.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use thiserror::Error;

mod properties;

pub use properties::{
    AxisOrder, Contiguity, DataType, DeviceClass, InferenceError, Mutability, OperatorProperties,
    OperatorStage, PropertyAnnotations, PropertyInference, Representation, Residency, ShapeDim,
    ValueGranularity, ValueProperties, ValueShape,
};

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
}

/// Stable logical node category. Domain details live in the erased payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Source,
    Index,
    Op,
    Batch,
    Cache,
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
            let label = node
                .payload
                .as_ref()
                .map(|p| format!(" {}::{}", p.domain(), p.name()))
                .unwrap_or_default();
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
    fn replacement_keeps_id_and_validation_rejects_cycle() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(source).unwrap();
        plan.replace_node(source, LogicalNode::new(NodeKind::Op, [source], None))
            .unwrap();
        assert_eq!(plan.validate(), Err(PlanError::Cycle(source.index())));
    }
}
