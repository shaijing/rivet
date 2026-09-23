//! Explicit vision domain hooks for Rivet's generic planner.

use std::sync::Arc;

use rivet_plan::{
    AxisOrder, FusionCandidate, FusionRule, LogicalPlan, OptimizerContext, OptimizerPass,
    PassResult, PlanPlugin, PlanRegistry, PropertyAnnotations, PropertyInference, ValueProperties,
};

use super::inference::VisionPropertyInference;
use super::op::ImageOp;
use crate::errors::invalid_pipeline;
use crate::sample::image::ImageAxisOrder;

pub(crate) fn optimize_vision_plan(
    plan: &mut LogicalPlan,
    workers: usize,
) -> Result<OptimizerContext, crate::errors::VisionError> {
    let mut registry = PlanRegistry::default();
    registry.register_plugin(&VisionPlanPlugin { workers });
    rivet_plan::optimize(plan, &registry)
        .map(|(context, _)| context)
        .map_err(|error| invalid_pipeline(format!("logical optimizer failed: {error}")))
}

struct VisionPlanPlugin {
    workers: usize,
}
impl PlanPlugin for VisionPlanPlugin {
    fn name(&self) -> &'static str {
        "vision"
    }
    fn property_inference(&self) -> Option<Arc<dyn PropertyInference>> {
        Some(Arc::new(VisionPropertyInference::new(self.workers)))
    }
    fn optimizer_passes(&self) -> Vec<Arc<dyn OptimizerPass>> {
        let inference: Arc<dyn PropertyInference> =
            Arc::new(VisionPropertyInference::new(self.workers));
        vec![
            Arc::new(Canonicalize),
            Arc::new(InferProperties(inference)),
            Arc::new(EliminateIdentityLayout),
            Arc::new(EliminateDeadNodes),
            Arc::new(InferProperties(Arc::new(VisionPropertyInference::new(
                self.workers,
            )))),
            Arc::new(SemanticRewriteBoundary),
            Arc::new(FusionDiscovery),
            Arc::new(PlacementBoundary),
        ]
    }
    fn fusion_rules(&self) -> Vec<Arc<dyn FusionRule>> {
        vec![Arc::new(NormalizeLayoutFusion {
            workers: self.workers,
        })]
    }
}

struct EliminateDeadNodes;
impl OptimizerPass for EliminateDeadNodes {
    fn name(&self) -> &'static str {
        "dead-node-elimination"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let mapping = plan
            .prune_unreachable()
            .map_err(|error| error.to_string())?;
        let changed = mapping.iter().any(Option::is_none)
            || mapping
                .iter()
                .enumerate()
                .any(|(old, new)| new.is_some_and(|new| new.index() != old));
        if changed {
            context.clear_annotations();
        }
        Ok(PassResult {
            changed,
            diagnostics: Vec::new(),
        })
    }
}

struct SemanticRewriteBoundary;
impl OptimizerPass for SemanticRewriteBoundary {
    fn name(&self) -> &'static str {
        "semantic-rewrite"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        _context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        plan.validate().map_err(|error| error.to_string())?;
        Ok(PassResult::unchanged())
    }
}

struct Canonicalize;
impl OptimizerPass for Canonicalize {
    fn name(&self) -> &'static str {
        "canonicalization"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        _context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        plan.validate().map_err(|error| error.to_string())?;
        Ok(PassResult::unchanged())
    }
}

struct InferProperties(Arc<dyn PropertyInference>);
impl OptimizerPass for InferProperties {
    fn name(&self) -> &'static str {
        "property-inference-validation"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let properties = plan
            .infer_properties(self.0.as_ref())
            .map_err(|error| error.to_string())?;
        context.set_annotations(properties);
        Ok(PassResult::unchanged())
    }
}

struct EliminateIdentityLayout;
impl OptimizerPass for EliminateIdentityLayout {
    fn name(&self) -> &'static str {
        "simplify-identity-layout"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let order = plan.preorder().map_err(|error| error.to_string())?;
        let mut changed = false;
        for id in order {
            let node = plan.node(id).map_err(|error| error.to_string())?;
            let Some(ImageOp::Layout(config)) = node.payload_as::<ImageOp>() else {
                continue;
            };
            let Some(input) = node.inputs().get(0) else {
                continue;
            };
            let same_order =
                context.annotations().get(input).and_then(axis_order) == Some(config.axis_order);
            if same_order {
                plan.redirect_uses(id, input)
                    .map_err(|error| error.to_string())?;
                context.diagnostics.push(rivet_plan::Diagnostic {
                    code: "rewrite.identity-layout",
                    message: format!("removed redundant Layout at node %{}", id.index()),
                });
                changed = true;
            }
        }
        if changed {
            context.clear_annotations();
        }
        Ok(PassResult {
            changed,
            diagnostics: Vec::new(),
        })
    }
}

fn axis_order(properties: &ValueProperties) -> Option<ImageAxisOrder> {
    match properties.axis_order.as_ref()? {
        AxisOrder::Hwc | AxisOrder::Nhwc => Some(ImageAxisOrder::Hwc),
        AxisOrder::Chw | AxisOrder::Nchw => Some(ImageAxisOrder::Chw),
        AxisOrder::Other(_) => None,
    }
}

/// The marker pass establishes the required ordering point. Registered domain
/// rules are invoked by rivet-plan immediately after this pass.
struct FusionDiscovery;
impl OptimizerPass for FusionDiscovery {
    fn name(&self) -> &'static str {
        "fusion-discovery"
    }
    fn run(
        &self,
        _plan: &mut LogicalPlan,
        _context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        Ok(PassResult::unchanged())
    }
}

struct NormalizeLayoutFusion {
    workers: usize,
}
impl FusionRule for NormalizeLayoutFusion {
    fn name(&self) -> &'static str {
        "normalize-layout"
    }
    fn discover(
        &self,
        plan: &LogicalPlan,
        annotations: &PropertyAnnotations,
    ) -> Result<Vec<FusionCandidate>, String> {
        let mut candidates = Vec::new();
        for layout_id in plan.preorder().map_err(|error| error.to_string())? {
            let layout = plan.node(layout_id).map_err(|error| error.to_string())?;
            let Some(ImageOp::Layout(layout_config)) = layout.payload_as::<ImageOp>() else {
                continue;
            };
            let Some(normalize_id) = layout.inputs().get(0) else {
                continue;
            };
            let normalize = plan.node(normalize_id).map_err(|error| error.to_string())?;
            if !matches!(
                normalize.payload_as::<ImageOp>(),
                Some(ImageOp::Normalize(_))
            ) {
                continue;
            }
            let Some(input_id) = normalize.inputs().get(0) else {
                continue;
            };
            let props = annotations.get(input_id);
            let legal = props.is_some_and(|p| {
                p.dtype == Some(rivet_plan::DataType::U8)
                    && axis_order(p) == Some(ImageAxisOrder::Hwc)
                    && layout_config.axis_order == ImageAxisOrder::Chw
                    && !(self.workers > 0
                        && p.operator.is_some_and(|operator| {
                            operator.sample_stage_has_work
                                && operator.stage != rivet_plan::OperatorStage::Batch
                        }))
            });
            let reason = if legal {
                "requires U8 HWC input, CHW output, and batch-stage Normalize; properties satisfy the rule".to_owned()
            } else {
                "requires U8 HWC input, CHW output, and batch-stage Normalize; inferred properties do not satisfy the rule"
                    .to_owned()
            };
            candidates.push(FusionCandidate {
                rule: self.name(),
                nodes: vec![normalize_id, layout_id],
                name: "NormalizeToChw".to_owned(),
                legal,
                reason,
            });
        }
        Ok(candidates)
    }
}

struct PlacementBoundary;
impl OptimizerPass for PlacementBoundary {
    fn name(&self) -> &'static str {
        "placement-boundary"
    }
    fn run(
        &self,
        _plan: &mut LogicalPlan,
        _context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        Ok(PassResult::unchanged().diagnostic(
            "placement.deferred",
            "physical placement is handled by the existing CPU compiler",
        ))
    }
}
