use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use thiserror::Error;

use crate::{DomainId, LogicalNode, NodeId, PlanError};

/// Opaque domain-specific value properties carried alongside common planning
/// facts. Implementations stay in their domain crate so adding a new domain
/// never requires extending a central property enum.
pub trait DomainProperties: std::any::Any + fmt::Debug + Send + Sync {
    fn domain_id(&self) -> DomainId;
    fn as_any(&self) -> &dyn std::any::Any;
    fn equals(&self, other: &dyn DomainProperties) -> bool;
}

/// Convenient type-erased wrapper for a domain's own comparable property type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainPropertiesValue<T> {
    domain_id: DomainId,
    value: T,
}

impl<T> DomainPropertiesValue<T> {
    pub fn new(domain_id: DomainId, value: T) -> Self {
        Self { domain_id, value }
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T> DomainProperties for DomainPropertiesValue<T>
where
    T: std::any::Any + fmt::Debug + PartialEq + Eq + Send + Sync,
{
    fn domain_id(&self) -> DomainId {
        self.domain_id
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn equals(&self, other: &dyn DomainProperties) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

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
#[derive(Clone)]
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
    pub domain: Option<Arc<dyn DomainProperties>>,
}

impl Default for ValueProperties {
    fn default() -> Self {
        Self {
            representation: None,
            dtype: None,
            shape: None,
            axis_order: None,
            contiguity: None,
            residency: None,
            granularity: None,
            mutability: None,
            operator: None,
            domain: None,
        }
    }
}

impl fmt::Debug for ValueProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValueProperties")
            .field("representation", &self.representation)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("axis_order", &self.axis_order)
            .field("contiguity", &self.contiguity)
            .field("residency", &self.residency)
            .field("granularity", &self.granularity)
            .field("mutability", &self.mutability)
            .field("operator", &self.operator)
            .field("domain", &self.domain)
            .finish()
    }
}

impl PartialEq for ValueProperties {
    fn eq(&self, other: &Self) -> bool {
        self.representation == other.representation
            && self.dtype == other.dtype
            && self.shape == other.shape
            && self.axis_order == other.axis_order
            && self.contiguity == other.contiguity
            && self.residency == other.residency
            && self.granularity == other.granularity
            && self.mutability == other.mutability
            && self.operator == other.operator
            && match (&self.domain, &other.domain) {
                (None, None) => true,
                (Some(lhs), Some(rhs)) => lhs.domain_id() == rhs.domain_id() && lhs.equals(&**rhs),
                _ => false,
            }
    }
}

impl Eq for ValueProperties {}

impl ValueProperties {
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn with_shape(mut self, dims: impl IntoIterator<Item = ShapeDim>) -> Self {
        self.shape = Some(ValueShape(dims.into_iter().collect()));
        self
    }

    pub fn with_domain(mut self, properties: impl DomainProperties + 'static) -> Self {
        self.domain = Some(Arc::new(properties));
        self
    }

    pub fn domain_as<T: std::any::Any>(&self) -> Option<&T> {
        self.domain.as_ref()?.as_any().downcast_ref()
    }
}

/// Domain supplied property inference hook. `inputs` follow the node's ordered
/// input edges; source nodes receive an empty slice.
pub trait PropertyInference: Send + Sync {
    /// Domain served by this inference provider. `None` is reserved for a
    /// registry-wide provider that intentionally handles every domain.
    fn domain_id(&self) -> Option<DomainId> {
        None
    }

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
