use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use thiserror::Error;

use crate::{LogicalNode, NodeId, PlanError};

/// Semantic representation carried by a pipeline edge.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Representation {
    EncodedImage,
    Image,
    Tensor,
    Bytes,
    Scalar,
    Other(Arc<str>),
}

/// Backend-independent scalar element type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    BF16,
    F16,
    F32,
    F64,
    Bool,
    Other(Arc<str>),
}

/// A dimension whose extent may be known only after reading a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShapeDim {
    Known(usize),
    Dynamic,
}

/// Shape can be unknown by rank, or known-rank with dynamic dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueShape(pub Vec<ShapeDim>);

impl ValueShape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[ShapeDim] {
        &self.0
    }
}

/// Semantic axis labels; independent from physical contiguity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AxisOrder {
    Hwc,
    Chw,
    Nhwc,
    Nchw,
    Other(Arc<[Arc<str>]>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Contiguity {
    Contiguous,
    Strided,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceClass {
    Cpu,
    Cuda,
    Metal,
    Other(Arc<str>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Residency {
    Host,
    Device(DeviceClass),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mutability {
    Immutable,
    Mutable,
    Unknown,
}

/// Value granularity at this point in a logical pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueGranularity {
    Sample,
    Batch,
    Stream,
    Unknown,
}

/// Coarse logical execution barrier, kept separate from value granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorStage {
    Source,
    Sample,
    Batch,
}

/// Compatibility metadata used while lowering a logical pipeline to the
/// current sample/batch execution model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OperatorProperties {
    pub stage: OperatorStage,
    pub sample_stage_has_work: bool,
}

/// Properties inferred for a logical value. `None` means unknown.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValueProperties {
    pub representation: Option<Representation>,
    pub dtype: Option<DataType>,
    pub shape: Option<ValueShape>,
    pub axis_order: Option<AxisOrder>,
    pub contiguity: Option<Contiguity>,
    pub residency: Option<Residency>,
    pub granularity: Option<ValueGranularity>,
    pub mutability: Option<Mutability>,
    pub operator: Option<OperatorProperties>,
}

impl ValueProperties {
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn with_shape(mut self, dims: impl IntoIterator<Item = ShapeDim>) -> Self {
        self.shape = Some(ValueShape(dims.into_iter().collect()));
        self
    }
}

/// Domain supplied property inference hook. `inputs` follow the node's ordered
/// input edges; source nodes receive an empty slice.
pub trait PropertyInference: Send + Sync {
    fn infer_node(
        &self,
        node: &LogicalNode,
        inputs: &[ValueProperties],
    ) -> Result<ValueProperties, String>;
}

/// Results of a logical plan property inference pass.
#[derive(Clone, Debug, Default)]
pub struct PropertyAnnotations {
    pub(crate) values: HashMap<NodeId, ValueProperties>,
}

impl PropertyAnnotations {
    pub fn get(&self, id: NodeId) -> Option<&ValueProperties> {
        self.values.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &ValueProperties)> {
        self.values.iter().map(|(id, properties)| (*id, properties))
    }

    pub fn insert(&mut self, id: NodeId, properties: ValueProperties) -> Option<ValueProperties> {
        self.values.insert(id, properties)
    }

    pub fn remove(&mut self, id: NodeId) -> Option<ValueProperties> {
        self.values.remove(&id)
    }
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("invalid logical plan: {0}")]
    InvalidPlan(#[from] PlanError),
    #[error("property inference failed at node %{}: {message}", .node.index())]
    Node { node: NodeId, message: String },
    #[error("property inference found a cycle at node %{}", .0.index())]
    Cycle(NodeId),
}

impl fmt::Display for ValueGranularity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sample => "sample",
            Self::Batch => "batch",
            Self::Stream => "stream",
            Self::Unknown => "unknown",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LogicalNode, LogicalPlan, NodeKind};

    struct PassThrough;

    impl PropertyInference for PassThrough {
        fn infer_node(
            &self,
            _node: &LogicalNode,
            inputs: &[ValueProperties],
        ) -> Result<ValueProperties, String> {
            Ok(inputs.first().cloned().unwrap_or_else(|| ValueProperties {
                representation: Some(Representation::Image),
                granularity: Some(ValueGranularity::Sample),
                ..ValueProperties::unknown()
            }))
        }
    }

    #[test]
    fn inference_walks_dependencies_before_consumers() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let op = plan.add_node(LogicalNode::new(NodeKind::Op, [source], None));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [op], None));
        plan.set_root(sink).unwrap();

        let properties = plan.infer_properties(&PassThrough).unwrap();
        assert_eq!(
            properties.get(source).unwrap().representation,
            Some(Representation::Image)
        );
        assert_eq!(
            properties.get(sink).unwrap().granularity,
            Some(ValueGranularity::Sample)
        );
        assert_eq!(properties.iter().count(), 3);
    }
}
