//! Explicit vision domain hooks for Rivet's generic planner.

use std::sync::Arc;

use rivet_plan::{
    AxisOrder, Contiguity, DeviceClass, FusionCandidate, FusionRule, KernelCapabilities,
    KernelCapability, KernelClass, KernelRequirements, LogicalNode, LogicalPlan, MachineProfile,
    NodeKind, OperatorStage, OptimizerContext, OptimizerPass, PassResult, PhysicalCandidate,
    PhysicalCandidateProvider, PlanPlugin, PlanRegistry, PropertyAnnotations, PropertyInference,
    ValueProperties,
};

use super::inference::VisionPropertyInference;
use super::logical::FusionGroupPayload;
use super::op::{ImageOp, IndexOp, SourceOp};
use crate::errors::invalid_pipeline;
use crate::sample::image::ImageAxisOrder;

#[cfg(test)]
pub(crate) fn optimize_vision_plan(
    plan: &mut LogicalPlan,
    workers: usize,
) -> Result<OptimizerContext, crate::errors::VisionError> {
    optimize_vision_plan_with_placement(plan, workers).map(|(context, _)| context)
}

pub(crate) fn optimize_vision_plan_with_placement(
    plan: &mut LogicalPlan,
    workers: usize,
) -> Result<(OptimizerContext, rivet_plan::PlacementPlan), crate::errors::VisionError> {
    let machine = vision_machine_profile(workers);
    let mut registry = PlanRegistry::default();
    registry.register_plugin(&VisionPlanPlugin {
        workers,
        enable_cuda_augmentation_fusion: false,
        machine: machine.clone(),
    });
    let (context, _) = rivet_plan::optimize(plan, &registry)
        .map_err(|error| invalid_pipeline(format!("logical optimizer failed: {error}")))?;
    let placement = rivet_plan::place(
        plan,
        context.annotations(),
        &vision_kernel_capabilities(),
        &machine,
    )
    .map_err(|error| invalid_pipeline(format!("physical placement failed: {error}")))?;
    Ok((context, placement))
}
fn vision_machine_profile(workers: usize) -> MachineProfile {
    MachineProfile {
        cpu_threads: workers.max(1),
        ..MachineProfile::default()
    }
}

impl super::builder::ImagePipeline {
    /// Produce a deterministic placement explanation for this pipeline on the
    /// supplied machine profile. Only registered implementations are eligible.
    pub fn placement_explain(
        &self,
        mut machine: MachineProfile,
    ) -> Result<String, crate::errors::VisionError> {
        if !self.device_cuts.is_empty() {
            return Err(invalid_pipeline(
                "DeviceCut placement validation is not implemented yet",
            ));
        }
        // A backend's registered capabilities describe what could execute on
        // that device; they do not opt a pipeline into device execution.
        machine.available_devices = vec![DeviceClass::Cpu];
        let mut plan = self.to_logical_plan();
        let enable_cuda_augmentation_fusion =
            machine.available_devices.contains(&DeviceClass::Cuda)
                && has_fixed_shape_batch_source(&plan);
        let mut registry = PlanRegistry::default();
        registry.register_plugin(&VisionPlanPlugin {
            workers: self.runtime.num_workers,
            enable_cuda_augmentation_fusion,
            machine,
        });
        let (context, _) = rivet_plan::optimize(&mut plan, &registry)
            .map_err(|error| invalid_pipeline(format!("logical optimizer failed: {error}")))?;
        let mut explanation = context
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code == "placement.explain")
            .map(|diagnostic| diagnostic.message.clone())
            .ok_or_else(|| invalid_pipeline("optimizer did not produce placement explanation"))?;
        for (id, node) in plan.nodes() {
            for candidate in registry.physical_candidates_for(node, context.annotations().get(id)) {
                explanation.push_str(&format!(
                    "  physical candidate %{}: {} backend={}\n",
                    id.index(),
                    candidate.name,
                    candidate.backend
                ));
            }
        }
        Ok(explanation)
    }
}

fn has_fixed_shape_batch_source(plan: &LogicalPlan) -> bool {
    plan.nodes().any(|(_, node)| {
        node.payload_as::<SourceOp>()
            .is_some_and(SourceOp::supports_batch_read)
    })
}

struct VisionPlanPlugin {
    workers: usize,
    enable_cuda_augmentation_fusion: bool,
    machine: MachineProfile,
}
impl PlanPlugin for VisionPlanPlugin {
    fn name(&self) -> &'static str {
        "vision"
    }
    fn domain_id(&self) -> Option<rivet_plan::DomainId> {
        Some(crate::VISION_DOMAIN_ID)
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
            Arc::new(InferProperties(Arc::new(VisionPropertyInference::new(
                self.workers,
            )))),
            Arc::new(SemanticRewrite {
                workers: self.workers,
            }),
            Arc::new(EliminateDeadNodes),
            Arc::new(InferProperties(Arc::new(VisionPropertyInference::new(
                self.workers,
            )))),
            Arc::new(IndexSourcePushdown {
                workers: self.workers,
            }),
            Arc::new(FusionDiscovery),
            Arc::new(EliminateDeadNodes),
            Arc::new(InferProperties(Arc::new(VisionPropertyInference::new(
                self.workers,
            )))),
            Arc::new(PlacementBoundary {
                capabilities: vision_kernel_capabilities(),
                machine: self.machine.clone(),
            }),
        ]
    }
    fn fusion_rules(&self) -> Vec<Arc<dyn FusionRule>> {
        vec![Arc::new(NormalizeLayoutFusion {
            workers: self.workers,
            enable_cuda_augmentation_fusion: self.enable_cuda_augmentation_fusion,
        })]
    }
    fn physical_candidates(&self) -> Vec<Arc<dyn PhysicalCandidateProvider>> {
        vec![Arc::new(VisionPhysicalCandidates)]
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
            ..PassResult::default()
        })
    }
}

struct SemanticRewrite {
    workers: usize,
}
impl OptimizerPass for SemanticRewrite {
    fn name(&self) -> &'static str {
        "semantic-rewrite"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let mut changed = false;
        let order = plan.preorder().map_err(|error| error.to_string())?;
        for id in order {
            let node = plan.node(id).map_err(|error| error.to_string())?;
            let Some(input_id) = node.inputs().get(0) else {
                continue;
            };
            if let Some(crop) = node.payload_as::<ImageOp>() {
                let input_properties = context.annotations().get(input_id);
                if input_properties.is_some_and(|properties| is_identity_crop(crop, properties)) {
                    plan.redirect_uses(id, input_id)
                        .map_err(|error| error.to_string())?;
                    context.diagnostics.push(rivet_plan::Diagnostic {
                        code: "rewrite.identity-crop",
                        message: format!(
                            "removed full-image crop at node %{} using known input shape; values, layout, dtype, and randomness are unchanged",
                            id.index()
                        ),
                    });
                    changed = true;
                    continue;
                }
            }
            let Some(ImageOp::ConvertImageDtype(config)) = node.payload_as::<ImageOp>() else {
                continue;
            };
            let Some(input_properties) = context.annotations().get(input_id) else {
                continue;
            };

            // A conversion that preserves dtype is a semantic no-op. Dtype is
            // the only changed property for this operation; shape, layout,
            // ordering, randomness, and values are unchanged.
            let target_dtype = match config.dtype {
                rivet_core::DType::U8 => rivet_plan::DataType::U8,
                rivet_core::DType::F32 => rivet_plan::DataType::F32,
                _ => continue,
            };
            if input_properties.dtype == Some(target_dtype) {
                // Keep this node as a batch-stage barrier when worker sample
                // work precedes it; removing it could move following batch
                // ops onto worker lanes.
                if self.workers > 0
                    && input_properties.operator.is_some_and(|operator| {
                        operator.sample_stage_has_work
                            && operator.stage != rivet_plan::OperatorStage::Batch
                    })
                {
                    continue;
                }
                plan.redirect_uses(id, input_id)
                    .map_err(|error| error.to_string())?;
                context.diagnostics.push(rivet_plan::Diagnostic {
                    code: "rewrite.identity-dtype",
                    message: format!("removed identity ConvertImageDtype at node %{}", id.index()),
                });
                changed = true;
                continue;
            }

            // U8 -> F32 conversion followed by Normalize computes the same
            // `u8 / 255 -> (x - mean) / std` values as Normalize's U8 kernel.
            // Require the conversion to feed only this Normalize so bypassing
            // it cannot alter a sibling consumer that expects F32 values.
            if config.dtype != rivet_core::DType::F32
                || input_properties.dtype != Some(rivet_plan::DataType::U8)
            {
                continue;
            }
            // Keep Normalize on the same side of the worker stage barrier.
            if self.workers > 0
                && input_properties.operator.is_some_and(|operator| {
                    operator.sample_stage_has_work
                        && operator.stage != rivet_plan::OperatorStage::Batch
                })
            {
                continue;
            }
            let children = plan.children(id).map_err(|error| error.to_string())?;
            let [normalize_id] = children.as_slice() else {
                continue;
            };
            let normalize = plan
                .node(*normalize_id)
                .map_err(|error| error.to_string())?;
            if !matches!(
                normalize.payload_as::<ImageOp>(),
                Some(ImageOp::Normalize(_))
            ) || normalize.inputs().get(0) != Some(id)
            {
                continue;
            }
            plan.redirect_uses(id, input_id)
                .map_err(|error| error.to_string())?;
            context.diagnostics.push(rivet_plan::Diagnostic {
                code: "rewrite.dtype-late-promotion",
                message: format!(
                    "bypassed U8->F32 conversion at node %{} before Normalize; preserves U8 HWC/CHW order and uses Normalize's equivalent U8 scaling path",
                    id.index()
                ),
            });
            changed = true;
        }
        if changed {
            context.clear_annotations();
        }
        Ok(PassResult {
            changed,
            diagnostics: Vec::new(),
            ..PassResult::default()
        })
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
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let mapping = plan.canonicalize().map_err(|error| error.to_string())?;
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
            ..PassResult::default()
        })
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
            ..PassResult::default()
        })
    }
}

/// Push selection toward source reads using approved, adjacent IR rewrites.
/// Index order and image order are each preserved. Unknown operators remain
/// barriers, following the rule-local approach of Polars SlicePushDown.
struct IndexSourcePushdown {
    workers: usize,
}
impl OptimizerPass for IndexSourcePushdown {
    fn name(&self) -> &'static str {
        "source-index-pushdown"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let order = plan
            .topological_order()
            .map_err(|error| error.to_string())?;
        let mut changed = false;
        let mut moved = 0usize;
        let mut index_count = 0usize;
        for id in order {
            if plan.node(id).map_err(|error| error.to_string())?.kind() != NodeKind::Index {
                continue;
            }
            index_count += 1;
            loop {
                let index = plan.node(id).map_err(|error| error.to_string())?.clone();
                if index.payload_as::<IndexOp>().is_none() || index.inputs().len() != 1 {
                    return Err(format!(
                        "index node %{} must carry a unary vision IndexOp",
                        id.index()
                    ));
                }
                let input_id = index.inputs().get(0).expect("unary index");
                let input = plan
                    .node(input_id)
                    .map_err(|error| error.to_string())?
                    .clone();
                if matches!(input.kind(), NodeKind::Source | NodeKind::Index) {
                    // Adjacent indexes are intentionally never commuted: e.g.
                    // Shuffle -> Take selects different samples from Take -> Shuffle.
                    break;
                }
                let rejection = if input.inputs().len() != 1 {
                    Some("multi-input operation is a selection pushdown barrier")
                } else {
                    let consumers = plan
                        .topological_order()
                        .map_err(|error| error.to_string())?
                        .into_iter()
                        .map(|consumer| {
                            plan.node(consumer)
                                .expect("validated node")
                                .inputs()
                                .iter()
                                .filter(|dependency| *dependency == input_id)
                                .count()
                        })
                        .sum::<usize>();
                    if consumers != 1 || plan.root().ok() == Some(input_id) {
                        Some("shared operation is a selection pushdown barrier")
                    } else {
                        super::semantics::allows_index_pushdown(
                            &input,
                            input
                                .inputs()
                                .get(0)
                                .and_then(|upstream| context.annotations().get(upstream)),
                        )
                        .err()
                    }
                };
                if let Some(reason) = rejection {
                    let diagnostic = rivet_plan::Diagnostic {
                        code: "rewrite.index-pushdown-blocked",
                        message: format!(
                            "retained index %{} after %{}: {reason}",
                            id.index(),
                            input_id.index()
                        ),
                    };
                    if !context.diagnostics.contains(&diagnostic) {
                        context.diagnostics.push(diagnostic);
                    }
                    break;
                }
                let upstream = input.inputs().get(0).expect("checked unary image");
                // A -> Image -> Index -> consumers becomes A -> Index -> Image
                // -> consumers. Redirect first while Image does not yet use Index,
                // so global rewiring cannot create an Index self-reference.
                plan.redirect_uses(id, input_id)
                    .map_err(|error| error.to_string())?;
                plan.replace_node(id, index.with_inputs([upstream]))
                    .map_err(|error| error.to_string())?;
                plan.replace_node(input_id, input.with_inputs([id]))
                    .map_err(|error| error.to_string())?;
                context.clear_annotations();
                plan.validate().map_err(|error| error.to_string())?;
                let annotations = plan
                    .infer_properties(&VisionPropertyInference::new(self.workers))
                    .map_err(|error| error.to_string())?;
                context.set_annotations(annotations);
                changed = true;
                moved += 1;
            }
        }
        let mut result = PassResult {
            changed,
            ..PassResult::default()
        };
        if moved > 0 {
            result = result.diagnostic(
                "rewrite.index-source-pushdown",
                format!("moved selection across {moved} independent image operation(s); image order, index order, and stable source-sample random streams are preserved; unselected sample errors are not evaluated"),
            );
        } else if index_count > 0 {
            // Keep an explanation for plans already built with a source prefix.
            let diagnostic = rivet_plan::Diagnostic {
                code: "rewrite.index-source-prefix",
                message: format!(
                    "{index_count} index operation(s) retain declared order; only a contiguous source prefix is eligible for sampler lowering"
                ),
            };
            if !context.diagnostics.contains(&diagnostic) {
                context.diagnostics.push(diagnostic);
            }
        }
        Ok(result)
    }
}

fn axis_order(properties: &ValueProperties) -> Option<ImageAxisOrder> {
    match properties.axis_order.as_ref()? {
        AxisOrder::Hwc | AxisOrder::Nhwc => Some(ImageAxisOrder::Hwc),
        AxisOrder::Chw | AxisOrder::Nchw => Some(ImageAxisOrder::Chw),
        AxisOrder::Other(_) => None,
    }
}

fn is_identity_crop(op: &ImageOp, properties: &ValueProperties) -> bool {
    let Some(shape) = properties.shape.as_ref() else {
        return false;
    };
    let dims = shape.dims();
    if dims.len() != 3 {
        return false;
    }
    let spatial_dims = match axis_order(properties) {
        Some(ImageAxisOrder::Hwc) => (dims[0], dims[1]),
        Some(ImageAxisOrder::Chw) => (dims[1], dims[2]),
        None => return false,
    };
    let (rivet_plan::ShapeDim::Known(height), rivet_plan::ShapeDim::Known(width)) = spatial_dims
    else {
        return false;
    };
    match op {
        ImageOp::Crop(config) => {
            config.x == 0
                && config.y == 0
                && config.width as usize == width
                && config.height as usize == height
        }
        ImageOp::CenterCrop(config) => {
            config.width as usize == width && config.height as usize == height
        }
        _ => false,
    }
}

/// The marker pass establishes the required ordering point. Registered domain
/// rules are invoked by rivet-plan immediately after this pass.
struct FusionDiscovery;
impl OptimizerPass for FusionDiscovery {
    fn name(&self) -> &'static str {
        "fusion-discovery"
    }
    fn is_fusion_discovery(&self) -> bool {
        true
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
    enable_cuda_augmentation_fusion: bool,
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
        for id in plan.preorder().map_err(|error| error.to_string())? {
            let node = plan.node(id).map_err(|error| error.to_string())?;
            if let Some(ImageOp::Layout(layout_config)) = node.payload_as::<ImageOp>() {
                let Some(normalize_id) = node.inputs().get(0) else {
                    continue;
                };
                let normalize = plan.node(normalize_id).map_err(|error| error.to_string())?;
                if !matches!(
                    normalize.payload_as::<ImageOp>(),
                    Some(ImageOp::Normalize(_))
                ) {
                    continue;
                }
                let Some(normalize_input) = normalize.inputs().get(0) else {
                    continue;
                };
                let mut augmentation_ids = Vec::new();
                let mut input_id = normalize_input;
                if self.enable_cuda_augmentation_fusion {
                    loop {
                        let input_node = plan.node(input_id).map_err(|error| error.to_string())?;
                        if !input_node
                            .payload_as::<ImageOp>()
                            .is_some_and(is_cuda_batch_augmentation)
                        {
                            break;
                        }
                        augmentation_ids.push(input_id);
                        let Some(previous) = input_node.inputs().get(0) else {
                            break;
                        };
                        input_id = previous;
                    }
                    augmentation_ids.reverse();
                }
                let mut crop_count = 0;
                for augmentation_id in &augmentation_ids {
                    let augmentation = plan
                        .node(*augmentation_id)
                        .map_err(|error| error.to_string())?;
                    crop_count += usize::from(matches!(
                        augmentation.payload_as::<ImageOp>(),
                        Some(ImageOp::RandomResizedCrop(_))
                    ));
                }
                if crop_count > 1 {
                    augmentation_ids.clear();
                    input_id = normalize_input;
                }
                let props = annotations.get(input_id);
                let allow_sample_work =
                    self.enable_cuda_augmentation_fusion && !augmentation_ids.is_empty();
                let legal = props.is_some_and(|p| {
                    p.dtype == Some(rivet_plan::DataType::U8)
                        && axis_order(p) == Some(ImageAxisOrder::Hwc)
                        && layout_config.axis_order == ImageAxisOrder::Chw
                        && (allow_sample_work
                            || !(self.workers > 0
                                && p.operator.is_some_and(|operator| {
                                    operator.sample_stage_has_work
                                        && operator.stage != rivet_plan::OperatorStage::Batch
                                })))
                });
                let reason = if legal {
                    "requires U8 HWC input, CHW output, and batch-stage Normalize; properties satisfy the rule".to_owned()
                } else {
                    "requires U8 HWC input, CHW output, and batch-stage Normalize; inferred properties do not satisfy the rule"
                        .to_owned()
                };
                candidates.push(FusionCandidate {
                    rule: self.name(),
                    nodes: augmentation_ids
                        .into_iter()
                        .chain([normalize_id, id])
                        .collect(),
                    name: "NormalizeToChw".to_owned(),
                    legal,
                    reason,
                });
            }

            // Crop followed by resize cannot be reordered unchanged: crop
            // coordinates are in source pixels, while resize coordinates and
            // interpolation borders are relative to the cropped image.
            if matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Resize(_))) {
                if let Some(crop_id) = node.inputs().get(0) {
                    let crop = plan.node(crop_id).map_err(|error| error.to_string())?;
                    if matches!(
                        crop.payload_as::<ImageOp>(),
                        Some(ImageOp::Crop(_) | ImageOp::CenterCrop(_))
                    ) {
                        candidates.push(FusionCandidate {
                            rule: self.name(),
                            nodes: vec![crop_id, id],
                            name: "CropResize".to_owned(),
                            legal: false,
                            reason: "not safe to reorder without transforming the crop rectangle and preserving interpolation/border semantics; no crop-resize fused kernel is registered".to_owned(),
                        });
                    }
                }
            }

            // A horizontal flip commutes with per-channel affine normalization
            // for the same HWC image values, but the available implementation
            // crosses sample and batch stages, so it remains unfused.
            if !self.enable_cuda_augmentation_fusion
                && matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Normalize(_)))
            {
                if let Some(flip_id) = node.inputs().get(0) {
                    let flip = plan.node(flip_id).map_err(|error| error.to_string())?;
                    if matches!(flip.payload_as::<ImageOp>(), Some(ImageOp::Flip(_))) {
                        let input_id = flip.inputs().get(0);
                        let props = input_id.and_then(|input| annotations.get(input));
                        let semantic_commute = props.is_some_and(|p| {
                            p.dtype == Some(rivet_plan::DataType::U8)
                                && axis_order(p) == Some(ImageAxisOrder::Hwc)
                        });
                        candidates.push(FusionCandidate {
                            rule: self.name(),
                            nodes: vec![flip_id, id],
                            name: "FlipNormalize".to_owned(),
                            legal: false,
                            reason: if semantic_commute {
                                "spatial flip commutes with per-channel affine normalization for U8 HWC values, but sample-to-batch stage movement changes worker execution and no cross-stage fused kernel is registered".to_owned()
                            } else {
                                "requires known U8 HWC input to establish commutation; properties are insufficient".to_owned()
                            },
                        });
                    }
                }
            }
        }
        Ok(candidates)
    }

    fn apply(
        &self,
        plan: &mut LogicalPlan,
        candidate: &FusionCandidate,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        if candidate.name != "NormalizeToChw" {
            return Ok(PassResult::unchanged());
        }
        if candidate.nodes.len() < 2 {
            return Err("NormalizeToChw fusion requires two node ids".to_owned());
        }
        let (augmentation_ids, tail) = candidate.nodes.split_at(candidate.nodes.len() - 2);
        let [normalize_id, layout_id] = tail else {
            unreachable!("candidate length was checked")
        };
        let normalize_node = plan
            .node(*normalize_id)
            .map_err(|error| error.to_string())?;
        let layout_node = plan.node(*layout_id).map_err(|error| error.to_string())?;
        let Some(ImageOp::Normalize(normalize)) = normalize_node.payload_as::<ImageOp>() else {
            return Err("NormalizeToChw fusion lost its Normalize node".to_owned());
        };
        let Some(ImageOp::Layout(layout)) = layout_node.payload_as::<ImageOp>() else {
            return Err("NormalizeToChw fusion lost its Layout node".to_owned());
        };
        let mut input = normalize_node
            .inputs()
            .get(0)
            .ok_or_else(|| "Normalize node has no input".to_owned())?;
        let mut ops = Vec::with_capacity(augmentation_ids.len() + 2);
        if let Some(first_augmentation_id) = augmentation_ids.first() {
            let first_augmentation = plan
                .node(*first_augmentation_id)
                .map_err(|error| error.to_string())?;
            input = first_augmentation
                .inputs()
                .get(0)
                .ok_or_else(|| "augmentation node has no input".to_owned())?;
        }
        for augmentation_id in augmentation_ids {
            let augmentation = plan
                .node(*augmentation_id)
                .map_err(|error| error.to_string())?;
            let Some(op) = augmentation.payload_as::<ImageOp>() else {
                return Err("NormalizeToChw fusion lost an augmentation node".to_owned());
            };
            ops.push(op.clone());
        }
        ops.push(ImageOp::Normalize(normalize.clone()));
        ops.push(ImageOp::Layout(*layout));
        let fused = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [input],
            Some(Arc::new(FusionGroupPayload {
                name: "NormalizeToChw",
                ops,
            })),
        ));
        plan.redirect_uses(*layout_id, fused)
            .map_err(|error| error.to_string())?;
        context.clear_annotations();
        Ok(PassResult::changed().diagnostic(
            "fusion.applied",
            format!(
                "replaced Normalize %{} + Layout %{} with NormalizeToChw FusionGroup %{}",
                normalize_id.index(),
                layout_id.index(),
                fused.index()
            ),
        ))
    }
}

fn is_cuda_batch_augmentation(op: &ImageOp) -> bool {
    matches!(
        op,
        ImageOp::Flip(_) | ImageOp::RandomHorizontalFlip(_) | ImageOp::RandomResizedCrop(_)
    )
}

/// Selection currently executes only as a contiguous, unbranched source
/// prefix. Keep residual selections in the IR and report the unsupported
/// runtime capability before placement emits an unrelated kernel error.
fn validate_source_index_prefix(plan: &LogicalPlan) -> Result<(), String> {
    for id in plan
        .topological_order()
        .map_err(|error| error.to_string())?
    {
        if plan.node(id).map_err(|error| error.to_string())?.kind() != NodeKind::Index {
            continue;
        }
        let mut current = id;
        let legal = loop {
            let node = plan.node(current).map_err(|error| error.to_string())?;
            if node.kind() != NodeKind::Index || node.inputs().len() != 1 {
                break false;
            }
            let input = node.inputs().get(0).expect("unary index");
            let children = plan.children(input).map_err(|error| error.to_string())?;
            if children.as_slice() != [current] {
                break false;
            }
            match plan.node(input).map_err(|error| error.to_string())?.kind() {
                NodeKind::Source => break true,
                NodeKind::Index => current = input,
                _ => break false,
            }
        };
        if !legal {
            return Err(format!(
                "index operation at node %{} remains outside the shared source prefix: branch-local or post-barrier index execution is not implemented; the optimizer retained its declared position",
                id.index()
            ));
        }
    }
    Ok(())
}

struct PlacementBoundary {
    capabilities: KernelCapabilities,
    machine: MachineProfile,
}

impl OptimizerPass for PlacementBoundary {
    fn name(&self) -> &'static str {
        "placement-boundary"
    }
    fn is_placement_boundary(&self) -> bool {
        true
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        validate_source_index_prefix(plan)?;
        let placement = rivet_plan::place(
            plan,
            context.annotations(),
            &self.capabilities,
            &self.machine,
        )
        .map_err(|error| error.to_string())?;
        let mut explanation = placement
            .explain(plan, context.annotations())
            .map_err(|error| error.to_string())?;
        for diagnostic in context.diagnostics.iter().filter(|diagnostic| {
            diagnostic.code.starts_with("fusion.") || diagnostic.code.starts_with("rewrite.")
        }) {
            explanation.push_str(&format!("  {}: {}\n", diagnostic.code, diagnostic.message));
        }
        if let Some(source) = plan
            .nodes()
            .find_map(|(_, node)| node.payload_as::<super::op::SourceOp>())
        {
            let capability = source.capabilities();
            explanation.push_str(&format!(
                "  source access={:?} batched={} zero_copy={} parallel={} async={} read_device={:?}\n",
                capability.access_pattern,
                capability.batched_reads,
                capability.zero_copy,
                capability.parallel_reads,
                capability.async_reads,
                capability.read_device,
            ));
        }
        Ok(PassResult::unchanged().diagnostic("placement.explain", explanation))
    }
}

struct VisionPhysicalCandidates;
impl PhysicalCandidateProvider for VisionPhysicalCandidates {
    fn candidates(
        &self,
        node: &LogicalNode,
        properties: Option<&ValueProperties>,
    ) -> Vec<PhysicalCandidate> {
        let Some(payload) = node.payload() else {
            if node.kind() == NodeKind::Sink {
                return vec![PhysicalCandidate {
                    name: "vision-output-sink".to_owned(),
                    backend: "cpu".to_owned(),
                }];
            }
            return Vec::new();
        };
        if payload.domain_id() != Some(crate::VISION_DOMAIN_ID) {
            return Vec::new();
        }
        let name = match (node.kind(), payload.name()) {
            (NodeKind::Source, "Source") => "vision-source-read".to_owned(),
            (NodeKind::Index, "IndexOp") => "vision-index-sampler".to_owned(),
            (NodeKind::Op, _) if payload.as_domain_op().is_some() => {
                // Defer concrete kernel compatibility to KernelCapabilities.
                if properties.is_none() {
                    return Vec::new();
                }
                "vision-image-cpu".to_owned()
            }
            (NodeKind::Op, "FusionGroup") => {
                let Some(group) = node.payload_as::<FusionGroupPayload>() else {
                    return Vec::new();
                };
                if group.name != "NormalizeToChw"
                    || properties.is_none_or(|properties| {
                        properties.dtype != Some(rivet_plan::DataType::F32)
                            || axis_order(properties) != Some(ImageAxisOrder::Chw)
                    })
                {
                    return Vec::new();
                }
                "vision-normalize-to-chw-fused".to_owned()
            }
            (NodeKind::Batch, "BatchConfig") => "vision-batch-assembly".to_owned(),
            _ => return Vec::new(),
        };
        let candidates = vec![PhysicalCandidate {
            name,
            backend: "cpu".to_owned(),
        }];
        #[cfg(feature = "cuda")]
        if node
            .payload_as::<FusionGroupPayload>()
            .is_some_and(|group| group.name == "NormalizeToChw")
        {
            let mut candidates = candidates;
            candidates.push(PhysicalCandidate {
                name: "vision-normalize-to-chw-fused".to_owned(),
                backend: "cuda".to_owned(),
            });
            return candidates;
        }
        candidates
    }
}

pub(super) fn vision_kernel_capabilities() -> KernelCapabilities {
    use rivet_plan::NodeKind;

    let mut caps = KernelCapabilities::default();
    let cpu = DeviceClass::Cpu;
    let cpu_alignment = rivet_core::CPU_STORAGE_ALIGNMENT;
    let mut register = |operator: &str,
                        node_kind,
                        class,
                        stage,
                        parallel,
                        in_place,
                        contiguity,
                        fusion_tags: &[&str]| {
        let display_operator = operator
            .strip_suffix("::*")
            .map(|domain| format!("{domain}::ImageOp"))
            .unwrap_or_else(|| operator.to_owned());
        caps.register(KernelCapability {
            name: format!("{display_operator}-cpu-{stage:?}"),
            operator: operator.to_owned(),
            node_kind,
            device: cpu.clone(),
            class,
            requirements: KernelRequirements {
                output_stage: Some(stage),
                ..KernelRequirements::default()
            },
            fusion_tags: fusion_tags.iter().map(|tag| (*tag).to_owned()).collect(),
            alignment_bytes: cpu_alignment,
            contiguity,
            temporary_bytes: 0,
            cost_hint: rivet_plan::KernelCostHint::default(),
            in_place,
            parallel,
        });
    };
    register(
        "vision::Source",
        NodeKind::Source,
        KernelClass::Source,
        OperatorStage::Source,
        false,
        true,
        Contiguity::Unknown,
        &[],
    );
    register(
        "vision::IndexOp",
        NodeKind::Index,
        KernelClass::Source,
        OperatorStage::Source,
        false,
        true,
        Contiguity::Unknown,
        &[],
    );
    register(
        "vision::*",
        NodeKind::Op,
        KernelClass::Sample,
        OperatorStage::Sample,
        true,
        false,
        Contiguity::Unknown,
        &[],
    );
    register(
        "vision::*",
        NodeKind::Op,
        KernelClass::Batch,
        OperatorStage::Batch,
        false,
        false,
        Contiguity::Unknown,
        &["Normalize", "Layout"],
    );
    register(
        "vision::*",
        NodeKind::Op,
        KernelClass::Sample,
        OperatorStage::Source,
        false,
        true,
        Contiguity::Unknown,
        &[],
    );
    register(
        "vision::BatchConfig",
        NodeKind::Batch,
        KernelClass::Batch,
        OperatorStage::Batch,
        false,
        false,
        Contiguity::Contiguous,
        &[],
    );
    for stage in [
        OperatorStage::Source,
        OperatorStage::Sample,
        OperatorStage::Batch,
    ] {
        register(
            "rivet::Sink",
            NodeKind::Sink,
            KernelClass::Sink,
            stage,
            false,
            true,
            Contiguity::Unknown,
            &[],
        );
    }
    drop(register);
    caps.register(KernelCapability {
        name: "vision::NormalizeToChw-cpu-Fused".to_owned(),
        operator: "vision::FusionGroup".to_owned(),
        node_kind: NodeKind::Op,
        device: cpu,
        class: KernelClass::Fused,
        requirements: KernelRequirements {
            input_representation: Some(rivet_plan::Representation::Image),
            input_dtype: Some(rivet_plan::DataType::U8),
            input_axis_order: Some(AxisOrder::Hwc),
            input_granularity: Some(rivet_plan::ValueGranularity::Sample),
            input_residency: Some(rivet_plan::Residency::Host),
            output_representation: Some(rivet_plan::Representation::Image),
            output_dtype: Some(rivet_plan::DataType::F32),
            output_axis_order: Some(AxisOrder::Chw),
            output_granularity: Some(rivet_plan::ValueGranularity::Sample),
            output_residency: Some(rivet_plan::Residency::Host),
            output_contiguity: Some(Contiguity::Contiguous),
            output_stage: Some(OperatorStage::Batch),
            ..KernelRequirements::default()
        },
        fusion_tags: vec!["NormalizeToChw".to_owned()],
        alignment_bytes: cpu_alignment,
        contiguity: Contiguity::Contiguous,
        temporary_bytes: 0,
        cost_hint: rivet_plan::KernelCostHint::default(),
        in_place: false,
        parallel: false,
    });
    let mut batch_fused = caps
        .iter()
        .find(|capability| capability.name == "vision::NormalizeToChw-cpu-Fused")
        .expect("registered CPU fusion")
        .clone();
    batch_fused.name = "vision::NormalizeToChw-cpu-BatchFused".to_owned();
    batch_fused.requirements.input_granularity = Some(rivet_plan::ValueGranularity::Batch);
    batch_fused.requirements.output_granularity = Some(rivet_plan::ValueGranularity::Batch);
    batch_fused.requirements.input_axis_order = Some(AxisOrder::Nhwc);
    batch_fused.requirements.output_axis_order = Some(AxisOrder::Nchw);
    caps.register(batch_fused);
    #[cfg(feature = "cuda")]
    {
        let mut cuda_fused = caps
            .iter()
            .find(|capability| capability.name == "vision::NormalizeToChw-cpu-Fused")
            .expect("CPU fused capability was registered above")
            .clone();
        cuda_fused.name = "vision::NormalizeToChw-cuda-Fused".to_owned();
        cuda_fused.device = DeviceClass::Cuda;
        caps.register(cuda_fused);
    }
    caps
}

#[cfg(test)]
mod device_cut_tests {
    use super::*;
    use crate::pipeline::op::ImageOp;
    use rivet_plan::{DeviceCut, DeviceTarget, LogicalNode};

    #[test]
    fn normalize_layout_fusion_does_not_cross_a_device_cut() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let normalize = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(ImageOp::normalize(vec![0.5; 3], vec![0.5; 3]))),
        ));
        let cut = plan.add_node(LogicalNode::new(
            NodeKind::DeviceCut,
            [normalize],
            Some(Arc::new(DeviceCut::new(DeviceTarget::cuda(0)))),
        ));
        let layout = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [cut],
            Some(Arc::new(ImageOp::hwc_to_chw())),
        ));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [layout], None));
        plan.set_root(sink).unwrap();

        let candidates = NormalizeLayoutFusion {
            workers: 0,
            enable_cuda_augmentation_fusion: true,
        }
        .discover(&plan, &PropertyAnnotations::default())
        .unwrap();
        assert!(candidates.is_empty());
        assert!(plan.validate().is_ok());
    }
}

#[cfg(test)]
mod index_pushdown_tests {
    use super::*;
    use crate::sample::image::DecodedSample;
    use crate::source::ImageSource;
    use rivet_data::dataset::Dataset;

    struct UnreadDataset;
    impl Dataset for UnreadDataset {
        type Item = DecodedSample;
        fn len(&self) -> usize {
            10
        }
        fn get_many(&self, _: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            panic!("IR optimization must not read source data")
        }
    }

    fn source(plan: &mut LogicalPlan) -> rivet_plan::NodeId {
        plan.add_node(LogicalNode::new(
            NodeKind::Source,
            [],
            Some(Arc::new(SourceOp::new(ImageSource::from_decoded(
                Arc::new(UnreadDataset),
            )))),
        ))
    }
    fn inferred_context(plan: &LogicalPlan) -> OptimizerContext {
        let mut context = OptimizerContext::default();
        context.set_annotations(
            plan.infer_properties(&VisionPropertyInference::new(0))
                .unwrap(),
        );
        context
    }

    #[test]
    fn adjacent_rewrites_preserve_both_sequences_and_random_identity() {
        let mut plan = LogicalPlan::new();
        let source = source(&mut plan);
        let crop = plan.add_node(
            LogicalNode::new(
                NodeKind::Op,
                [source],
                Some(Arc::new(ImageOp::random_crop(2, 2, 4))),
            )
            .with_semantic_identity(44),
        );
        let normalize = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [crop],
            Some(Arc::new(ImageOp::normalize(vec![0.5; 3], vec![0.5; 3]))),
        ));
        let take = plan.add_node(LogicalNode::new(
            NodeKind::Index,
            [normalize],
            Some(Arc::new(IndexOp::Take { count: 4 })),
        ));
        let shuffle = plan.add_node(LogicalNode::new(
            NodeKind::Index,
            [take],
            Some(Arc::new(IndexOp::Shuffle { seed: 11 })),
        ));
        let batch = plan.add_node(LogicalNode::new(
            NodeKind::Batch,
            [shuffle],
            Some(Arc::new(super::super::op::BatchConfig::new(4, false))),
        ));
        plan.set_root(batch).unwrap();
        let mut context = inferred_context(&plan);
        let pass = IndexSourcePushdown { workers: 0 };
        assert!(pass.run(&mut plan, &mut context).unwrap().changed);
        assert_eq!(
            plan.topological_order().unwrap(),
            [source, take, shuffle, crop, normalize, batch]
        );
        assert_eq!(plan.node(crop).unwrap().semantic_identity(), Some(44));
        assert!(plan.validate().is_ok());
        assert!(!pass.run(&mut plan, &mut context).unwrap().changed);
        assert!(context.annotations().get(crop).is_some());
    }

    #[test]
    fn random_operation_without_stable_identity_stops_selection() {
        let mut plan = LogicalPlan::new();
        let source = source(&mut plan);
        let crop = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(ImageOp::random_crop(2, 2, 4))),
        ));
        let take = plan.add_node(LogicalNode::new(
            NodeKind::Index,
            [crop],
            Some(Arc::new(IndexOp::Take { count: 4 })),
        ));
        plan.set_root(take).unwrap();
        let mut context = inferred_context(&plan);
        assert!(
            !IndexSourcePushdown { workers: 0 }
                .run(&mut plan, &mut context)
                .unwrap()
                .changed
        );
        assert_eq!(plan.node(take).unwrap().inputs().get(0), Some(crop));
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("stable semantic identity"))
        );
    }

    #[test]
    fn shared_transform_and_batch_are_pushdown_barriers() {
        let mut plan = LogicalPlan::new();
        let source = source(&mut plan);
        let image = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(ImageOp::invert())),
        ));
        let take = plan.add_node(LogicalNode::new(
            NodeKind::Index,
            [image],
            Some(Arc::new(IndexOp::Take { count: 4 })),
        ));
        let join = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [take, image],
            Some(Arc::new(super::super::ImageConcat { axis: 2 })),
        ));
        plan.set_root(join).unwrap();
        let mut context = inferred_context(&plan);
        assert!(
            !IndexSourcePushdown { workers: 0 }
                .run(&mut plan, &mut context)
                .unwrap()
                .changed
        );
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("shared operation"))
        );
        let batch = plan.add_node(LogicalNode::new(
            NodeKind::Batch,
            [image],
            Some(Arc::new(super::super::op::BatchConfig::new(4, false))),
        ));
        let take_batch = plan.add_node(LogicalNode::new(
            NodeKind::Index,
            [batch],
            Some(Arc::new(IndexOp::Take { count: 1 })),
        ));
        plan.set_root(take_batch).unwrap();
        let mut context = inferred_context(&plan);
        assert!(
            !IndexSourcePushdown { workers: 0 }
                .run(&mut plan, &mut context)
                .unwrap()
                .changed
        );
        assert_eq!(plan.node(take_batch).unwrap().inputs().get(0), Some(batch));
    }

    #[test]
    fn unknown_unary_payload_is_not_inferred_to_be_pure() {
        let mut plan = LogicalPlan::new();
        let source = source(&mut plan);
        let unknown = plan.add_node(LogicalNode::new(NodeKind::Op, [source], None));
        let take = plan.add_node(LogicalNode::new(
            NodeKind::Index,
            [unknown],
            Some(Arc::new(IndexOp::Take { count: 4 })),
        ));
        plan.set_root(take).unwrap();
        let mut context = OptimizerContext::default();
        assert!(
            !IndexSourcePushdown { workers: 0 }
                .run(&mut plan, &mut context)
                .unwrap()
                .changed
        );
        assert_eq!(plan.node(take).unwrap().inputs().get(0), Some(unknown));
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("unknown or multi-input"))
        );
    }
}
