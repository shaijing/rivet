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
#[cfg(feature = "cuda")]
use super::op::BatchConfig;
use super::op::{ImageOp, IndexOp, SourceOp};
use crate::errors::invalid_pipeline;
use crate::sample::image::ImageAxisOrder;

#[cfg(test)]
pub(crate) fn optimize_vision_plan(
    plan: &mut LogicalPlan,
    workers: usize,
) -> Result<OptimizerContext, crate::errors::VisionError> {
    optimize_vision_plan_for_sink(plan, workers, None).map(|(context, _)| context)
}

pub(crate) fn optimize_vision_plan_for_sink(
    plan: &mut LogicalPlan,
    workers: usize,
    sink_device_ordinal: Option<usize>,
) -> Result<(OptimizerContext, rivet_plan::PlacementPlan), crate::errors::VisionError> {
    let machine = vision_machine_profile(workers, sink_device_ordinal.is_some());
    let enable_cuda_augmentation_fusion =
        sink_device_ordinal.is_some() && has_fixed_shape_batch_source(plan);
    let mut registry = PlanRegistry::default();
    registry.register_plugin(&VisionPlanPlugin {
        workers,
        enable_cuda_augmentation_fusion,
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
    #[cfg(feature = "cuda")]
    let placement = {
        let mut placement = placement;
        if sink_device_ordinal.is_some() {
            place_fused_vision_batch_on_cuda(plan, context.annotations(), &machine, &mut placement);
        }
        placement
    };
    Ok((context, placement))
}

#[cfg(feature = "cuda")]
fn place_fused_vision_batch_on_cuda(
    plan: &LogicalPlan,
    annotations: &PropertyAnnotations,
    machine: &MachineProfile,
    placement: &mut rivet_plan::PlacementPlan,
) {
    let Some((id, node)) = plan.nodes().find(|(_, node)| {
        node.payload_as::<FusionGroupPayload>()
            .is_some_and(|group| group.name == "NormalizeToChw")
    }) else {
        return;
    };
    let Some(input) = node.inputs().get(0).and_then(|id| annotations.get(id)) else {
        return;
    };
    let Some(output) = annotations.get(id) else {
        return;
    };
    if input.dtype != Some(rivet_plan::DataType::U8)
        || input.axis_order != Some(AxisOrder::Hwc)
        || output.dtype != Some(rivet_plan::DataType::F32)
        || output.axis_order != Some(AxisOrder::Chw)
    {
        return;
    }
    let Some(candidate) = placement
        .candidates
        .iter_mut()
        .find(|candidate| candidate.node == id)
    else {
        return;
    };
    let batch_size = plan
        .nodes()
        .find_map(|(_, node)| node.payload_as::<BatchConfig>().map(|batch| batch.size))
        .unwrap_or(1) as u64;
    let input_bytes_per_sample = estimate_property_bytes(input, machine.unknown_value_bytes);
    let transfer_bytes = input_bytes_per_sample.saturating_mul(batch_size);
    let output_bytes =
        estimate_property_bytes(output, machine.unknown_value_bytes).saturating_mul(batch_size);
    let mut cost = candidate.cost.clone();
    cost.host_bytes = 0;
    cost.device_bytes = output_bytes;
    cost.transfer_bytes = transfer_bytes;
    cost.launch_count = 1;
    cost.synchronization_count = 0;
    cost.compute_score = output_bytes.max(1) as f64 / 1_000_000_000.0;
    cost.total_score = cost.compute_score
        + transfer_bytes as f64 / machine.host_to_device_bytes_per_sec.max(1) as f64
        + cost.allocation_count as f64 * 0.000_001
        + machine.kernel_launch_seconds;
    candidate.kernel = "vision::NormalizeToChw-cuda-Fused".to_owned();
    candidate.device = DeviceClass::Cuda;
    candidate.cost = cost;
    candidate.alternatives.push(format!(
        "vision::NormalizeToChw-cuda-Fused@Cuda={:.6}",
        candidate.cost.total_score
    ));
    placement.transfer_boundaries.clear();
    placement
        .transfer_boundaries
        .push((id, DeviceClass::Cpu, DeviceClass::Cuda));
}

#[cfg(feature = "cuda")]
fn estimate_property_bytes(properties: &ValueProperties, fallback: u64) -> u64 {
    let Some(shape) = properties.shape.as_ref() else {
        return fallback;
    };
    let Some(elements) = shape.0.iter().try_fold(1u64, |count, dim| match dim {
        rivet_plan::ShapeDim::Known(value) => count.checked_mul(*value as u64),
        rivet_plan::ShapeDim::Dynamic => None,
    }) else {
        return fallback;
    };
    let element_bytes = match properties.dtype.as_ref() {
        Some(rivet_plan::DataType::U8 | rivet_plan::DataType::I8 | rivet_plan::DataType::Bool) => 1,
        Some(
            rivet_plan::DataType::U16
            | rivet_plan::DataType::I16
            | rivet_plan::DataType::F16
            | rivet_plan::DataType::BF16,
        ) => 2,
        Some(rivet_plan::DataType::U32 | rivet_plan::DataType::I32 | rivet_plan::DataType::F32) => {
            4
        }
        Some(rivet_plan::DataType::U64 | rivet_plan::DataType::I64 | rivet_plan::DataType::F64) => {
            8
        }
        _ => return fallback,
    };
    elements.saturating_mul(element_bytes)
}

fn vision_machine_profile(workers: usize, cuda_sink: bool) -> MachineProfile {
    let mut machine = MachineProfile {
        cpu_threads: workers.max(1),
        ..MachineProfile::default()
    };
    if cuda_sink {
        machine.available_devices.push(DeviceClass::Cuda);
        machine.preferred_sink_device = Some(DeviceClass::Cuda);
    }
    machine
}

impl super::builder::ImagePipeline {
    /// Produce a deterministic placement explanation for this pipeline on the
    /// supplied machine profile. Only registered implementations are eligible.
    pub fn placement_explain(
        &self,
        machine: MachineProfile,
    ) -> Result<String, crate::errors::VisionError> {
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
                let candidates = vec![PhysicalCandidate {
                    name: "vision-output-sink".to_owned(),
                    backend: "cpu".to_owned(),
                }];
                #[cfg(feature = "cuda")]
                let candidates = {
                    let mut candidates = candidates;
                    candidates.push(PhysicalCandidate {
                        name: "vision-output-sink-cuda".to_owned(),
                        backend: "cuda".to_owned(),
                    });
                    candidates
                };
                return candidates;
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
    #[cfg(feature = "cuda")]
    for stage in [
        OperatorStage::Source,
        OperatorStage::Sample,
        OperatorStage::Batch,
    ] {
        caps.register(KernelCapability {
            name: format!("vision-output-sink-cuda-{stage:?}"),
            operator: "rivet::Sink".to_owned(),
            node_kind: NodeKind::Sink,
            device: DeviceClass::Cuda,
            class: KernelClass::Sink,
            requirements: KernelRequirements {
                output_stage: Some(stage),
                ..KernelRequirements::default()
            },
            fusion_tags: Vec::new(),
            alignment_bytes: cpu_alignment,
            contiguity: Contiguity::Unknown,
            temporary_bytes: 0,
            in_place: true,
            parallel: false,
        });
    }
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
