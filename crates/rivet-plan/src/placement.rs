//! Static kernel capability matching, cost estimates, and device placement.

use thiserror::Error;

use crate::{
    AxisOrder, Contiguity, DataType, DeviceClass, LogicalPlan, Mutability, NodeId, NodeKind,
    OperatorStage, PropertyAnnotations, Representation, Residency, ShapeDim, ValueGranularity,
    ValueProperties,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelClass {
    Source,
    Sample,
    Batch,
    Fused,
    Sink,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelRequirements {
    pub input_representation: Option<Representation>,
    pub input_dtype: Option<DataType>,
    pub input_axis_order: Option<AxisOrder>,
    pub input_granularity: Option<ValueGranularity>,
    pub input_contiguity: Option<Contiguity>,
    pub input_residency: Option<Residency>,
    pub input_mutability: Option<Mutability>,
    pub input_stage: Option<OperatorStage>,
    pub output_representation: Option<Representation>,
    pub output_contiguity: Option<Contiguity>,
    pub output_residency: Option<Residency>,
    pub output_mutability: Option<Mutability>,
    pub output_stage: Option<OperatorStage>,
    pub output_dtype: Option<DataType>,
    pub output_axis_order: Option<AxisOrder>,
    pub output_granularity: Option<ValueGranularity>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCapability {
    pub name: String,
    /// Domain and payload name; `rivet::Sink` is used for a payload-free sink.
    pub operator: String,
    pub node_kind: NodeKind,
    pub device: DeviceClass,
    pub class: KernelClass,
    pub requirements: KernelRequirements,
    pub fusion_tags: Vec<String>,
    pub alignment_bytes: usize,
    pub contiguity: Contiguity,
    pub temporary_bytes: usize,
    pub in_place: bool,
    pub parallel: bool,
}

#[derive(Clone, Debug, Default)]
pub struct KernelCapabilities {
    kernels: Vec<KernelCapability>,
}

impl KernelCapabilities {
    pub fn register(&mut self, capability: KernelCapability) {
        self.kernels.push(capability);
    }
    pub fn iter(&self) -> impl Iterator<Item = &KernelCapability> {
        self.kernels.iter()
    }
    pub fn for_node<'a>(
        &'a self,
        plan: &LogicalPlan,
        id: NodeId,
    ) -> Result<Vec<&'a KernelCapability>, PlacementError> {
        let node = plan.node(id)?;
        let operator = match node.payload() {
            Some(payload) => format!("{}::{}", payload.domain(), payload.name()),
            None if node.kind() == NodeKind::Sink => "rivet::Sink".to_owned(),
            None => String::new(),
        };
        Ok(self
            .kernels
            .iter()
            .filter(|kernel| kernel.node_kind == node.kind() && kernel.operator == operator)
            .collect())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostEstimate {
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub transfer_bytes: u64,
    pub temporary_bytes: u64,
    pub allocation_count: u32,
    pub launch_count: u32,
    pub synchronization_count: u32,
    pub compute_score: f64,
    pub total_score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MachineProfile {
    pub cpu_threads: usize,
    pub available_devices: Vec<DeviceClass>,
    pub host_to_device_bytes_per_sec: u64,
    pub device_to_host_bytes_per_sec: u64,
    pub kernel_launch_seconds: f64,
    pub synchronization_seconds: f64,
    /// Conservative byte estimate used when shape or dtype is unknown.
    pub unknown_value_bytes: u64,
    pub preferred_sink_device: Option<DeviceClass>,
}

impl Default for MachineProfile {
    fn default() -> Self {
        Self {
            cpu_threads: 1,
            available_devices: vec![DeviceClass::Cpu],
            host_to_device_bytes_per_sec: 12_000_000_000,
            device_to_host_bytes_per_sec: 12_000_000_000,
            kernel_launch_seconds: 0.000_01,
            synchronization_seconds: 0.000_01,
            unknown_value_bytes: 1_048_576,
            preferred_sink_device: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacementCandidate {
    pub node: NodeId,
    pub kernel: String,
    pub device: DeviceClass,
    pub class: KernelClass,
    pub parallelism: usize,
    pub cost: CostEstimate,
    pub alternatives: Vec<String>,
    pub fusion_tags: Vec<String>,
    pub alignment_bytes: usize,
    pub contiguity: Contiguity,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlacementPlan {
    pub candidates: Vec<PlacementCandidate>,
    pub transfer_boundaries: Vec<(NodeId, DeviceClass, DeviceClass)>,
}

impl PlacementPlan {
    pub fn explain(
        &self,
        plan: &LogicalPlan,
        annotations: &PropertyAnnotations,
    ) -> Result<String, PlacementError> {
        let mut out = String::from("PlacementPlan\n");
        for selected in &self.candidates {
            let node = plan.node(selected.node)?;
            let props = annotations.get(selected.node);
            out.push_str(&format!(
                "  %{} {:?} kernel={} device={:?} parallelism={} cost={:.6} host={}B device={}B transfer={}B temp={}B alloc={} launch={} sync={} compute={:.3} alignment={} contiguity={:?} fusion={:?} candidates={:?} shape={:?} dtype={:?} layout={:?}\n",
                selected.node.index(), node.kind(), selected.kernel, selected.device,
                selected.parallelism, selected.cost.total_score, selected.cost.host_bytes,
                selected.cost.device_bytes, selected.cost.transfer_bytes,
                selected.cost.temporary_bytes,
                selected.cost.allocation_count, selected.cost.launch_count,
                selected.cost.synchronization_count, selected.cost.compute_score,
                selected.alignment_bytes, selected.contiguity, selected.fusion_tags, selected.alternatives,
                props.and_then(|p| p.shape.as_ref()), props.and_then(|p| p.dtype.as_ref()),
                props.and_then(|p| p.axis_order.as_ref()),
            ));
        }
        for (node, from, to) in &self.transfer_boundaries {
            out.push_str(&format!(
                "  transfer before %{}: {:?} -> {:?}\n",
                node.index(),
                from,
                to
            ));
        }
        Ok(out)
    }
}

#[derive(Debug, Error)]
pub enum PlacementError {
    #[error("invalid logical plan: {0}")]
    Plan(#[from] crate::PlanError),
    #[error("no registered compatible kernel for node %{node}: {reason}")]
    NoKernel { node: usize, reason: String },
    #[error("placement requires more than one CPU/device boundary")]
    TooManyTransferBoundaries,
    #[error("sink requires {device:?}, but no compatible registered kernel can produce it")]
    SinkDeviceUnavailable { device: DeviceClass },
}

/// Select only explicitly registered implementations. A small transition
/// penalty favors same-device regions; an optional sink constraint selects
/// the sink lane and records any required producer-to-sink transfer.
pub fn place(
    plan: &LogicalPlan,
    annotations: &PropertyAnnotations,
    capabilities: &KernelCapabilities,
    machine: &MachineProfile,
) -> Result<PlacementPlan, PlacementError> {
    plan.validate()?;
    let ids = plan.preorder()?.into_iter().rev().collect::<Vec<_>>();
    let mut output = PlacementPlan::default();
    let mut previous: Option<DeviceClass> = None;
    let mut boundary_count = 0;
    for (position, id) in ids.iter().copied().enumerate() {
        plan.node(id)?;
        let props = annotations.get(id);
        let available = capabilities.for_node(plan, id)?;
        let has_registered = !available.is_empty();
        let mut compatible = available
            .into_iter()
            .filter(|kernel| machine.available_devices.contains(&kernel.device))
            .filter(|kernel| properties_match(&kernel.requirements, plan, id, annotations))
            .filter(|kernel| {
                kernel.contiguity == Contiguity::Unknown
                    || props.is_some_and(|p| p.contiguity == Some(kernel.contiguity))
            })
            .collect::<Vec<_>>();
        if compatible.is_empty() {
            if position + 1 == ids.len() {
                if let Some(device) = &machine.preferred_sink_device {
                    return Err(PlacementError::SinkDeviceUnavailable {
                        device: device.clone(),
                    });
                }
            }
            return Err(PlacementError::NoKernel {
                node: id.index(),
                reason: if !has_registered {
                    "no implementation registered".to_owned()
                } else {
                    "registered kernels do not match properties or available devices".to_owned()
                },
            });
        }
        // A sink placement describes where its consumed value must reside.
        // Let the producer stay on its compatible lane; the explicit transfer
        // boundary below then accounts for a CPU -> device sink handoff.
        if plan.node(id)?.kind() == NodeKind::Sink {
            if let Some(sink_device) = &machine.preferred_sink_device {
                compatible.retain(|candidate| &candidate.device == sink_device);
                if compatible.is_empty() {
                    return Err(PlacementError::SinkDeviceUnavailable {
                        device: sink_device.clone(),
                    });
                }
            }
        }
        let alternatives = compatible
            .iter()
            .map(|candidate| {
                format!(
                    "{}@{:?}={:.6}",
                    candidate.name,
                    candidate.device,
                    candidate_score(candidate, props, machine, previous.as_ref())
                )
            })
            .collect::<Vec<_>>();
        let selected =
            compatible
                .into_iter()
                .min_by(|a, b| {
                    candidate_score(a, props, machine, previous.as_ref())
                        .total_cmp(&candidate_score(b, props, machine, previous.as_ref()))
                })
                .expect("nonempty compatible candidates");
        let parallelism = if selected.parallel {
            machine.cpu_threads.max(1)
        } else {
            1
        };
        let cost = estimate_cost(selected, props, parallelism, machine, previous.as_ref());
        if let Some(from) = previous.as_ref() {
            if from != &selected.device {
                boundary_count += 1;
                output
                    .transfer_boundaries
                    .push((id, from.clone(), selected.device.clone()));
            }
        }
        output.candidates.push(PlacementCandidate {
            node: id,
            kernel: selected.name.clone(),
            device: selected.device.clone(),
            class: selected.class,
            parallelism,
            cost,
            alternatives,
            fusion_tags: selected.fusion_tags.clone(),
            alignment_bytes: selected.alignment_bytes,
            contiguity: selected.contiguity,
        });
        previous = Some(selected.device.clone());
    }
    if boundary_count > 1 {
        return Err(PlacementError::TooManyTransferBoundaries);
    }
    Ok(output)
}

fn properties_match(
    req: &KernelRequirements,
    plan: &LogicalPlan,
    id: NodeId,
    annotations: &PropertyAnnotations,
) -> bool {
    let node = match plan.node(id) {
        Ok(node) => node,
        Err(_) => return false,
    };
    let input = node
        .inputs()
        .get(0)
        .and_then(|input| annotations.get(input));
    let output = annotations.get(id);
    let matches_input = |actual: Option<&ValueProperties>| {
        actual.is_some_and(|p| {
            req.input_representation
                .as_ref()
                .is_none_or(|v| p.representation.as_ref() == Some(v))
                && req
                    .input_dtype
                    .as_ref()
                    .is_none_or(|v| p.dtype.as_ref() == Some(v))
                && req
                    .input_axis_order
                    .as_ref()
                    .is_none_or(|v| p.axis_order.as_ref() == Some(v))
                && req
                    .input_granularity
                    .is_none_or(|v| p.granularity == Some(v))
                && req.input_contiguity.is_none_or(|v| p.contiguity == Some(v))
                && req
                    .input_residency
                    .as_ref()
                    .is_none_or(|v| p.residency.as_ref() == Some(v))
                && req.input_mutability.is_none_or(|v| p.mutability == Some(v))
                && req
                    .input_stage
                    .is_none_or(|v| p.operator.is_some_and(|op| op.stage == v))
        })
    };
    let matches_output = output.is_some_and(|p| {
        req.output_representation
            .as_ref()
            .is_none_or(|v| p.representation.as_ref() == Some(v))
            && req
                .output_dtype
                .as_ref()
                .is_none_or(|v| p.dtype.as_ref() == Some(v))
            && req
                .output_axis_order
                .as_ref()
                .is_none_or(|v| p.axis_order.as_ref() == Some(v))
            && req
                .output_granularity
                .is_none_or(|v| p.granularity == Some(v))
            && req
                .output_contiguity
                .is_none_or(|v| p.contiguity == Some(v))
            && req
                .output_residency
                .as_ref()
                .is_none_or(|v| p.residency.as_ref() == Some(v))
            && req
                .output_mutability
                .is_none_or(|v| p.mutability == Some(v))
            && req
                .output_stage
                .is_none_or(|v| p.operator.is_some_and(|op| op.stage == v))
    });
    let has_input_requirements = req.input_representation.is_some()
        || req.input_dtype.is_some()
        || req.input_axis_order.is_some()
        || req.input_granularity.is_some()
        || req.input_contiguity.is_some()
        || req.input_residency.is_some()
        || req.input_mutability.is_some()
        || req.input_stage.is_some();
    (!has_input_requirements || matches_input(input)) && matches_output
}

fn candidate_score(
    kernel: &KernelCapability,
    props: Option<&ValueProperties>,
    machine: &MachineProfile,
    previous: Option<&DeviceClass>,
) -> f64 {
    estimate_cost(
        kernel,
        props,
        if kernel.parallel {
            machine.cpu_threads.max(1)
        } else {
            1
        },
        machine,
        previous,
    )
    .total_score
}

fn estimate_cost(
    kernel: &KernelCapability,
    props: Option<&ValueProperties>,
    parallelism: usize,
    machine: &MachineProfile,
    previous: Option<&DeviceClass>,
) -> CostEstimate {
    let temporary_bytes = kernel.temporary_bytes as u64;
    let bytes = props
        .and_then(byte_size)
        .unwrap_or(machine.unknown_value_bytes)
        .saturating_add(temporary_bytes);
    let device = !matches!(kernel.device, DeviceClass::Cpu);
    let transfer_bytes = if previous.is_some_and(|d| d != &kernel.device) {
        bytes
    } else {
        0
    };
    let allocation_count = u32::from(!kernel.in_place);
    let launch_count = u32::from(
        device && kernel.class != KernelClass::Source && kernel.class != KernelClass::Sink,
    );
    let synchronization_count = launch_count;
    let compute_score = (bytes.max(1) as f64) / 1_000_000_000.0 / (parallelism.max(1) as f64);
    let bandwidth = if previous.is_some_and(|d| !matches!(d, DeviceClass::Cpu)) {
        machine.device_to_host_bytes_per_sec
    } else {
        machine.host_to_device_bytes_per_sec
    };
    let transfer_score = if transfer_bytes == 0 {
        0.0
    } else {
        transfer_bytes as f64 / bandwidth.max(1) as f64
    };
    let total_score = compute_score
        + transfer_score
        + allocation_count as f64 * 0.000_001
        + launch_count as f64 * machine.kernel_launch_seconds
        + synchronization_count as f64 * machine.synchronization_seconds
        + if previous.is_some_and(|d| d != &kernel.device) {
            1_000_000_000.0
        } else {
            0.0
        };
    CostEstimate {
        host_bytes: if device { 0 } else { bytes },
        device_bytes: if device { bytes } else { 0 },
        transfer_bytes,
        temporary_bytes,
        allocation_count,
        launch_count,
        synchronization_count,
        compute_score,
        total_score,
    }
}

fn byte_size(properties: &ValueProperties) -> Option<u64> {
    let elements =
        properties
            .shape
            .as_ref()?
            .dims()
            .iter()
            .try_fold(1u64, |size, dim| match dim {
                ShapeDim::Known(value) => size.checked_mul(*value as u64),
                ShapeDim::Dynamic => None,
            })?;
    let width = match properties.dtype.as_ref()? {
        DataType::U8 | DataType::I8 | DataType::Bool => 1,
        DataType::U16 | DataType::I16 | DataType::F16 | DataType::BF16 => 2,
        DataType::U32 | DataType::I32 | DataType::F32 => 4,
        DataType::U64 | DataType::I64 | DataType::F64 => 8,
        DataType::Other(_) => return None,
    };
    elements.checked_mul(width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LogicalNode, PlanPayload};

    struct Payload;
    impl PlanPayload for Payload {
        fn domain(&self) -> &'static str {
            "demo"
        }
        fn name(&self) -> &'static str {
            "Op"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn unregistered_kernel_is_not_selected_and_explain_has_costs() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(
            NodeKind::Source,
            [],
            Some(std::sync::Arc::new(Payload)),
        ));
        plan.set_root(source).unwrap();
        let mut props = PropertyAnnotations::default();
        props.insert(
            source,
            ValueProperties {
                dtype: Some(DataType::U8),
                shape: Some(crate::ValueShape(vec![ShapeDim::Known(8)])),
                ..ValueProperties::default()
            },
        );
        let mut caps = KernelCapabilities::default();
        assert!(matches!(
            place(&plan, &props, &caps, &MachineProfile::default()),
            Err(PlacementError::NoKernel { .. })
        ));
        caps.register(KernelCapability {
            name: "demo-cpu".into(),
            operator: "demo::Op".into(),
            node_kind: NodeKind::Source,
            device: DeviceClass::Cpu,
            class: KernelClass::Source,
            requirements: KernelRequirements::default(),
            fusion_tags: Vec::new(),
            alignment_bytes: 1,
            contiguity: Contiguity::Unknown,
            temporary_bytes: 0,
            in_place: true,
            parallel: false,
        });
        let placement = place(&plan, &props, &caps, &MachineProfile::default()).unwrap();
        assert!(
            placement
                .explain(&plan, &props)
                .unwrap()
                .contains("compute=")
        );
    }
}
