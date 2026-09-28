//! Adapter between the public image builder and Rivet's domain-neutral IR.

use std::any::Any;
use std::sync::Arc;

use rivet_plan::{
    DeviceCut, DomainId, DomainOp, InferenceError, LogicalNode, LogicalPlan, NodeId, NodeKind,
    OpId, PlanError, PlanPayload, PropertyAnnotations, ValueProperties,
};

use super::builder::{ImagePipeline, PipelineStep};
use super::inference::VisionPropertyInference;
use super::op::{BatchConfig, ImageOp, IndexOp, SourceOp};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::runtime::RuntimeConfig;

#[derive(Clone, Copy)]
pub(crate) struct ImagePipelineContext {
    pub(crate) runtime: RuntimeConfig,
    pub(crate) epoch: u64,
    pub(crate) global_seed: Option<u64>,
}

impl PlanPayload for ImagePipelineContext {
    fn domain(&self) -> &'static str {
        "vision"
    }
    fn name(&self) -> &'static str {
        "ImagePipelineContext"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn domain_id(&self) -> Option<DomainId> {
        Some(crate::VISION_DOMAIN_ID)
    }
}

macro_rules! payload_impl {
    ($ty:ty, $name:literal) => {
        impl PlanPayload for $ty {
            fn domain(&self) -> &'static str {
                "vision"
            }
            fn name(&self) -> &'static str {
                $name
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn domain_id(&self) -> Option<DomainId> {
                Some(crate::VISION_DOMAIN_ID)
            }
        }
    };
}

payload_impl!(SourceOp, "Source");
payload_impl!(IndexOp, "IndexOp");
payload_impl!(BatchConfig, "BatchConfig");

impl PlanPayload for ImageOp {
    fn domain(&self) -> &'static str {
        "vision"
    }
    fn name(&self) -> &'static str {
        self.name()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn domain_id(&self) -> Option<DomainId> {
        Some(crate::VISION_DOMAIN_ID)
    }
    fn op_id(&self) -> Option<OpId> {
        Some(OpId::new(self.planning_op_id()))
    }
    fn as_domain_op(&self) -> Option<&dyn DomainOp> {
        Some(self)
    }
}

impl DomainOp for ImageOp {
    fn infer_properties(&self, inputs: &[ValueProperties]) -> Result<ValueProperties, String> {
        let [input] = inputs else {
            return Err(format!(
                "vision {} requires one input, got {}",
                self.name(),
                inputs.len()
            ));
        };
        super::inference::infer_image_op(self, input).map_err(|error| error.to_string())
    }
}

#[derive(Clone)]
pub(crate) struct FusionGroupPayload {
    pub(crate) name: &'static str,
    pub(crate) ops: Vec<ImageOp>,
}

impl PlanPayload for FusionGroupPayload {
    fn domain(&self) -> &'static str {
        "vision"
    }
    fn name(&self) -> &'static str {
        "FusionGroup"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn domain_id(&self) -> Option<DomainId> {
        Some(crate::VISION_DOMAIN_ID)
    }
}

impl ImagePipeline {
    /// Lower the builder's ordered operations into an arena-backed logical plan.
    pub fn to_logical_plan(&self) -> LogicalPlan {
        let mut plan = LogicalPlan::new();
        plan.set_context(Arc::new(ImagePipelineContext {
            runtime: self.runtime,
            epoch: self.epoch,
            global_seed: self.global_seed,
        }));
        let mut previous = plan.add_node(LogicalNode::new(
            NodeKind::Source,
            [],
            Some(Arc::new(self.source.clone())),
        ));
        let mut steps = self.steps.clone();
        // Retain compatibility with direct appends to the public operation
        // vectors. Builder methods always record their precise call order.
        for index in 0..self.index_ops.len() {
            if !steps
                .iter()
                .any(|step| matches!(step, PipelineStep::Index(i) if *i == index))
            {
                steps.push(PipelineStep::Index(index));
            }
        }
        for index in 0..self.ops.len() {
            if !steps
                .iter()
                .any(|step| matches!(step, PipelineStep::Image { index: i, .. } if *i == index))
            {
                steps.push(PipelineStep::Image {
                    index,
                    identity: None,
                });
            }
        }
        if !steps.iter().any(|step| matches!(step, PipelineStep::Batch)) && self.batch.is_some() {
            steps.push(PipelineStep::Batch);
        }
        let mut used_identities = steps
            .iter()
            .filter_map(|step| match step {
                PipelineStep::Image { identity, .. } => *identity,
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        let mut occurrences = std::collections::HashMap::<&'static str, u32>::new();
        for step in steps {
            let node = match step {
                PipelineStep::Index(index) => {
                    let Some(op) = self.index_ops.get(index) else {
                        continue;
                    };
                    LogicalNode::new(NodeKind::Index, [previous], Some(Arc::new(op.clone())))
                }
                PipelineStep::Image { index, identity } => {
                    let Some(op) = self.ops.get(index) else {
                        continue;
                    };
                    let mut node =
                        LogicalNode::new(NodeKind::Op, [previous], Some(Arc::new(op.clone())));
                    if let Some(kind) = op.random_key_kind() {
                        let occurrence = occurrences.entry(kind).or_default();
                        let key = if let Some(identity) = identity {
                            identity
                        } else {
                            loop {
                                let key = rivet_data::random::OpKey::from_parts(kind, *occurrence)
                                    .as_u64();
                                *occurrence += 1;
                                if used_identities.insert(key) {
                                    break key;
                                }
                            }
                        };
                        node = node.with_semantic_identity(key);
                    }
                    node
                }
                PipelineStep::Batch => {
                    // The marker owns position; the existing public field owns
                    // configuration, including direct replacement/removal.
                    let Some(batch) = self.batch else { continue };
                    LogicalNode::new(NodeKind::Batch, [previous], Some(Arc::new(batch)))
                }
                PipelineStep::DeviceCut(index) => {
                    let Some(cut) = self.device_cuts.get(index) else {
                        continue;
                    };
                    LogicalNode::new(
                        NodeKind::DeviceCut,
                        [previous],
                        Some(Arc::new(DeviceCut::new(cut.target.clone()))),
                    )
                }
            };
            previous = plan.add_node(node);
        }
        let root = plan.add_node(LogicalNode::new(NodeKind::Sink, [previous], None));
        plan.set_root(root)
            .expect("newly inserted logical root is valid");
        plan
    }

    /// Infer per-node representation, dtype, shape, layout, residency,
    /// granularity, and mutability without executing tensor operations.
    pub fn infer_properties(&self) -> RivetResult<PropertyAnnotations> {
        self.to_logical_plan()
            .infer_properties(&VisionPropertyInference::new(self.runtime.num_workers))
            .map_err(inference_error)
    }

    /// Restore an image builder from its logical representation.
    pub fn from_logical_plan(plan: &LogicalPlan) -> RivetResult<Self> {
        plan.validate().map_err(plan_error)?;
        let context = plan
            .context_as::<ImagePipelineContext>()
            .ok_or_else(|| invalid_pipeline("logical plan has no vision pipeline context"))?;

        let mut reversed = Vec::new();
        let mut current = plan.root().map_err(plan_error)?;
        loop {
            let node = plan.node(current).map_err(plan_error)?;
            reversed.push(current);
            if node.inputs().is_empty() {
                break;
            }
            if node.inputs().len() != 1 {
                return Err(invalid_pipeline(format!(
                    "vision lowering expects one input at node %{}",
                    current.index()
                )));
            }
            current = node.inputs().get(0).expect("one input was checked");
        }
        reversed.reverse();

        let mut steps = Vec::new();
        let mut source = None;
        let mut index_ops = Vec::new();
        let mut ops = Vec::new();
        let mut device_cuts = Vec::new();
        let mut batch = None;
        let last = reversed.last().copied();
        for id in reversed {
            let node = plan.node(id).map_err(plan_error)?;
            match node.kind() {
                NodeKind::Source if source.is_none() => {
                    source = Some(payload::<SourceOp>(node, id)?.clone());
                }
                NodeKind::Index => {
                    steps.push(PipelineStep::Index(index_ops.len()));
                    index_ops.push(payload::<IndexOp>(node, id)?.clone());
                }
                NodeKind::Op => {
                    if let Some(group) = node.payload_as::<FusionGroupPayload>() {
                        for op in &group.ops {
                            steps.push(PipelineStep::Image {
                                index: ops.len(),
                                identity: None,
                            });
                            ops.push(op.clone());
                        }
                    } else {
                        steps.push(PipelineStep::Image {
                            index: ops.len(),
                            identity: node.semantic_identity(),
                        });
                        ops.push(payload::<ImageOp>(node, id)?.clone());
                    }
                }
                NodeKind::DeviceCut => {
                    let cut = payload::<DeviceCut>(node, id)?;
                    steps.push(PipelineStep::DeviceCut(device_cuts.len()));
                    device_cuts.push(super::builder::DeviceCutPlacement {
                        after_ops: ops.len(),
                        after_batch: batch.is_some(),
                        target: cut.target.clone(),
                    });
                }
                NodeKind::Batch if batch.is_none() => {
                    let config = *payload::<BatchConfig>(node, id)?;
                    steps.push(PipelineStep::Batch);
                    batch = Some(config);
                }
                NodeKind::Sink if Some(id) == last => {}
                kind => {
                    return Err(invalid_pipeline(format!(
                        "unexpected {kind:?} node at %{} in vision logical plan",
                        id.index()
                    )));
                }
            }
        }
        let source = source.ok_or_else(|| invalid_pipeline("logical plan has no source node"))?;
        Ok(Self {
            steps,
            source,
            index_ops,
            ops,
            batch,
            device_cuts,
            runtime: context.runtime,
            epoch: context.epoch,
            global_seed: context.global_seed,
        })
    }
}

fn payload<T: Any>(node: &LogicalNode, id: NodeId) -> RivetResult<&T> {
    node.payload_as::<T>().ok_or_else(|| {
        invalid_pipeline(format!(
            "logical node %{} has an incompatible payload",
            id.index()
        ))
    })
}

fn plan_error(error: PlanError) -> crate::errors::VisionError {
    invalid_pipeline(format!("invalid logical plan: {error}"))
}

pub(crate) fn inference_error(error: InferenceError) -> crate::errors::VisionError {
    match error {
        InferenceError::Node { node, message } => invalid_pipeline(format!(
            "property inference failed at node %{}: {message}",
            node.index()
        )),
        other => invalid_pipeline(other.to_string()),
    }
}

/// Supply identities once, before any rewrite. Existing identities survive
/// edits and arena compaction; new manual graph operators receive deterministic
/// unused namespaces in the original graph's dependency order.
pub(crate) fn assign_missing_random_identities(plan: &mut LogicalPlan) -> RivetResult<()> {
    let order = plan.topological_order().map_err(plan_error)?;
    let mut used = std::collections::HashSet::new();
    for &id in &order {
        if let Some(identity) = plan.node(id).map_err(plan_error)?.semantic_identity() {
            used.insert(identity);
        }
    }
    let mut occurrences = std::collections::HashMap::<&'static str, u32>::new();
    for id in order {
        let node = plan.node(id).map_err(plan_error)?;
        let Some(kind) = node
            .payload_as::<ImageOp>()
            .and_then(ImageOp::random_key_kind)
        else {
            continue;
        };
        if node.semantic_identity().is_some() {
            continue;
        }
        let occurrence = occurrences.entry(kind).or_default();
        let identity = loop {
            let identity = rivet_data::random::OpKey::from_parts(kind, *occurrence).as_u64();
            *occurrence = occurrence
                .checked_add(1)
                .ok_or_else(|| invalid_pipeline("too many random operators"))?;
            if used.insert(identity) {
                break identity;
            }
        };
        plan.replace_node(id, node.clone().with_semantic_identity(identity))
            .map_err(plan_error)?;
    }
    Ok(())
}

#[cfg(test)]
mod ordered_builder_tests {
    use super::*;
    use crate::sample::image::EncodedImageSample;
    use rivet_data::dataset::Dataset;

    struct EmptyImages;
    impl Dataset for EmptyImages {
        type Item = EncodedImageSample;
        fn len(&self) -> usize {
            0
        }
        fn get_many(&self, _indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn public_batch_edits_change_configuration_without_moving_its_position() {
        let mut builder = ImagePipeline::new(Arc::new(EmptyImages))
            .decode_image()
            .batch(4, false)
            .normalize(vec![0.0; 3], vec![1.0; 3]);
        builder.batch = Some(BatchConfig::new(2, true));
        let plan = builder.to_logical_plan();
        let order = plan.topological_order().unwrap();
        let batch_position = order
            .iter()
            .position(|id| plan.node(*id).unwrap().kind() == NodeKind::Batch)
            .unwrap();
        let batch = plan
            .node(order[batch_position])
            .unwrap()
            .payload_as::<BatchConfig>()
            .unwrap();
        assert_eq!(batch.size, 2);
        assert!(batch.drop_last);
        assert_eq!(
            plan.node(order[batch_position + 1])
                .unwrap()
                .payload_as::<ImageOp>()
                .unwrap()
                .name(),
            "Normalize"
        );
        builder.batch = None;
        assert!(
            builder
                .to_logical_plan()
                .nodes()
                .all(|(_, node)| node.kind() != NodeKind::Batch)
        );
    }

    #[test]
    fn repeated_declared_batches_remain_visible_and_are_rejected() {
        let builder = ImagePipeline::new(Arc::new(EmptyImages))
            .decode_image()
            .batch(4, false)
            .batch(2, true);
        assert_eq!(
            builder
                .to_logical_plan()
                .nodes()
                .filter(|(_, node)| node.kind() == NodeKind::Batch)
                .count(),
            2
        );
        let error = match builder.compile() {
            Ok(_) => panic!("nested batching must not be silently discarded"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("batch requires sample granularity"),
            "{error}"
        );
    }
}
