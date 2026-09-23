//! Adapter between the public image builder and Rivet's domain-neutral IR.

use std::any::Any;
use std::sync::Arc;

use rivet_plan::{
    InferenceError, LogicalNode, LogicalPlan, NodeId, NodeKind, PlanError, PlanPayload,
    PropertyAnnotations,
};

use super::builder::ImagePipeline;
use super::inference::VisionPropertyInference;
use super::op::{BatchConfig, ImageOp, IndexOp, SourceOp};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::runtime::RuntimeConfig;

#[derive(Clone, Copy)]
struct ImagePipelineContext {
    runtime: RuntimeConfig,
    epoch: u64,
    global_seed: Option<u64>,
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
        }
    };
}

payload_impl!(SourceOp, "Source");
payload_impl!(IndexOp, "IndexOp");
payload_impl!(ImageOp, "ImageOp");
payload_impl!(BatchConfig, "BatchConfig");

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
        for op in &self.index_ops {
            previous = plan.add_node(LogicalNode::new(
                NodeKind::Index,
                [previous],
                Some(Arc::new(op.clone())),
            ));
        }
        for op in &self.ops {
            previous = plan.add_node(LogicalNode::new(
                NodeKind::Op,
                [previous],
                Some(Arc::new(op.clone())),
            ));
        }
        if let Some(batch) = self.batch {
            previous = plan.add_node(LogicalNode::new(
                NodeKind::Batch,
                [previous],
                Some(Arc::new(batch)),
            ));
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

        let mut source = None;
        let mut index_ops = Vec::new();
        let mut ops = Vec::new();
        let mut batch = None;
        let last = reversed.last().copied();
        for id in reversed {
            let node = plan.node(id).map_err(plan_error)?;
            match node.kind() {
                NodeKind::Source if source.is_none() => {
                    source = Some(payload::<SourceOp>(node, id)?.clone());
                }
                NodeKind::Index => index_ops.push(payload::<IndexOp>(node, id)?.clone()),
                NodeKind::Op => {
                    if let Some(group) = node.payload_as::<FusionGroupPayload>() {
                        ops.extend(group.ops.iter().cloned());
                    } else {
                        ops.push(payload::<ImageOp>(node, id)?.clone());
                    }
                }
                NodeKind::Batch if batch.is_none() => {
                    batch = Some(*payload::<BatchConfig>(node, id)?);
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
            source,
            index_ops,
            ops,
            batch,
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
