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
use super::op::{ImageOp, IndexOp};
use crate::errors::invalid_pipeline;
use crate::sample::image::ImageAxisOrder;

pub(crate) fn optimize_vision_plan(
    plan: &mut LogicalPlan,
    workers: usize,
) -> Result<OptimizerContext, crate::errors::VisionError> {
    let machine = MachineProfile {
        cpu_threads: workers.max(1),
        ..MachineProfile::default()
    };
    let mut registry = PlanRegistry::default();
    registry.register_plugin(&VisionPlanPlugin { workers, machine });
    rivet_plan::optimize(plan, &registry)
        .map(|(context, _)| context)
        .map_err(|error| invalid_pipeline(format!("logical optimizer failed: {error}")))
}

impl super::builder::ImagePipeline {
    /// Produce a deterministic placement explanation for this pipeline on the
    /// supplied machine profile. Only registered implementations are eligible.
    pub fn placement_explain(
        &self,
        machine: MachineProfile,
    ) -> Result<String, crate::errors::VisionError> {
        let mut plan = self.to_logical_plan();
        let mut registry = PlanRegistry::default();
        registry.register_plugin(&VisionPlanPlugin {
            workers: self.runtime.num_workers,
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

struct VisionPlanPlugin {
    workers: usize,
    machine: MachineProfile,
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
            Arc::new(IndexSourcePushdown),
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

/// Index operators are represented before sample transforms and compiled into
/// the sampler, so source reads already receive the selected indices only.
struct IndexSourcePushdown;
impl OptimizerPass for IndexSourcePushdown {
    fn name(&self) -> &'static str {
        "source-index-pushdown"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        _context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let mut saw_image_op = false;
        let mut index_count = 0usize;
        for id in plan
            .preorder()
            .map_err(|error| error.to_string())?
            .into_iter()
            .rev()
        {
            let node = plan.node(id).map_err(|error| error.to_string())?;
            match node.kind() {
                NodeKind::Op => saw_image_op = true,
                NodeKind::Index => {
                    if saw_image_op {
                        return Err(format!(
                            "index node %{} occurs after an image transform; source pushdown cannot preserve the declared plan order",
                            id.index()
                        ));
                    }
                    index_count += 1;
                    if node.payload_as::<IndexOp>().is_none() {
                        return Err(format!("index node %{} has no vision IndexOp", id.index()));
                    }
                }
                _ => {}
            }
        }
        let mut result = PassResult::unchanged();
        if index_count > 0 {
            result = result.diagnostic(
                "rewrite.index-source-pushdown",
                format!(
                    "{index_count} Skip/Take/Shuffle operation(s) are compiled into the source sampler before reads and image transforms"
                ),
            );
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
                    nodes: vec![normalize_id, id],
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
            if matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Normalize(_))) {
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
        let [normalize_id, layout_id] = candidate.nodes.as_slice() else {
            return Err("NormalizeToChw fusion requires two node ids".to_owned());
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
        let input = normalize_node
            .inputs()
            .get(0)
            .ok_or_else(|| "Normalize node has no input".to_owned())?;
        let fused = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [input],
            Some(Arc::new(FusionGroupPayload {
                name: "NormalizeToChw",
                ops: vec![
                    ImageOp::Normalize(normalize.clone()),
                    ImageOp::Layout(*layout),
                ],
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

struct PlacementBoundary {
    capabilities: KernelCapabilities,
    machine: MachineProfile,
}

impl OptimizerPass for PlacementBoundary {
    fn name(&self) -> &'static str {
        "placement-boundary"
    }
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
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
        for diagnostic in context
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code.starts_with("fusion."))
        {
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
        if payload.domain() != "vision" {
            return Vec::new();
        }
        let name = match (node.kind(), payload.name()) {
            (NodeKind::Source, "Source") => "vision-source-read".to_owned(),
            (NodeKind::Index, "IndexOp") => "vision-index-sampler".to_owned(),
            (NodeKind::Op, "ImageOp") => {
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
        vec![PhysicalCandidate {
            name,
            backend: "cpu".to_owned(),
        }]
    }
}

fn vision_kernel_capabilities() -> KernelCapabilities {
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
        caps.register(KernelCapability {
            name: format!("{operator}-cpu-{stage:?}"),
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
        "vision::ImageOp",
        NodeKind::Op,
        KernelClass::Sample,
        OperatorStage::Sample,
        true,
        false,
        Contiguity::Unknown,
        &[],
    );
    register(
        "vision::ImageOp",
        NodeKind::Op,
        KernelClass::Batch,
        OperatorStage::Batch,
        false,
        false,
        Contiguity::Unknown,
        &["Normalize", "Layout"],
    );
    register(
        "vision::ImageOp",
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
        in_place: false,
        parallel: false,
    });
    caps
}
