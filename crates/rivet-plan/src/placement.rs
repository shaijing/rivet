//! Static kernel capability matching, cost estimates, and device placement.

use std::collections::{BTreeSet, HashMap, HashSet};
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
    /// `domain::*` matches semantic `DomainOp` payloads in that domain.
    pub operator: String,
    pub node_kind: NodeKind,
    pub device: DeviceClass,
    pub class: KernelClass,
    pub requirements: KernelRequirements,
    pub fusion_tags: Vec<String>,
    pub alignment_bytes: usize,
    pub contiguity: Contiguity,
    pub temporary_bytes: usize,
    pub cost_hint: KernelCostHint,
    pub in_place: bool,
    pub parallel: bool,
}

/// Static traffic and work estimates supplied by the backend capability.
/// Pass counts describe full-value reads/writes; zero means the operation does
/// not access that side of the edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelCostHint {
    pub input_read_passes: u32,
    pub output_write_passes: u32,
    pub compute_ops_per_element: u32,
}

impl Default for KernelCostHint {
    fn default() -> Self {
        Self {
            input_read_passes: 1,
            output_write_passes: 1,
            compute_ops_per_element: 1,
        }
    }
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
        let (operator, domain_operator) = match node.payload() {
            Some(payload) => (
                format!("{}::{}", payload.domain(), payload.name()),
                payload
                    .as_domain_op()
                    .map(|_| format!("{}::*", payload.domain())),
            ),
            None if node.kind() == NodeKind::Sink => ("rivet::Sink".to_owned(), None),
            None => (String::new(), None),
        };
        Ok(self
            .kernels
            .iter()
            .filter(|kernel| {
                kernel.node_kind == node.kind()
                    && (kernel.operator == operator
                        || domain_operator
                            .as_ref()
                            .is_some_and(|wildcard| &kernel.operator == wildcard))
            })
            .collect())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostEstimate {
    /// Aggregate traffic summaries retained for compact explain output.
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub host_bytes_read: u64,
    pub host_bytes_written: u64,
    pub device_bytes_read: u64,
    pub device_bytes_written: u64,
    pub transfer_bytes: u64,
    pub temporary_bytes: u64,
    pub allocation_count: u32,
    pub launch_count: u32,
    pub synchronization_count: u32,
    pub compute_score: f64,
    pub transfer_score: f64,
    pub total_score: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceDescriptor {
    pub class: DeviceClass,
    pub ordinal: usize,
    pub memory_bytes: Option<u64>,
    pub async_copy: bool,
    pub supported_kernels: KernelSet,
}

/// Stable backend-independent identity for a device. Runtime handles and
/// stream identities are resolved after placement.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Accelerator { class: DeviceClass, ordinal: usize },
}

impl DeviceDescriptor {
    pub fn new(
        class: DeviceClass,
        ordinal: usize,
        memory_bytes: Option<u64>,
        async_copy: bool,
        supported_kernels: KernelSet,
    ) -> Self {
        Self {
            class,
            ordinal,
            memory_bytes,
            async_copy,
            supported_kernels,
        }
    }

    pub fn location(&self) -> DeviceLocation {
        if self.class == DeviceClass::Cpu {
            DeviceLocation::Cpu
        } else {
            DeviceLocation::Accelerator {
                class: self.class.clone(),
                ordinal: self.ordinal,
            }
        }
    }
}

/// Kernel names implemented by one concrete device. An empty set means that
/// the descriptor does not restrict kernels beyond the registered class
/// capabilities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelSet(BTreeSet<String>);

impl KernelSet {
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(names.into_iter().map(Into::into).collect())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

impl FromIterator<String> for KernelSet {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        Self::new(iter)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MachineProfile {
    pub cpu_threads: usize,
    /// Available device classes. CPU is always considered as the fallback
    /// implementation; this list primarily opts accelerator classes in.
    pub available_devices: Vec<DeviceClass>,
    /// Optional concrete accelerator inventory. Accelerator classes and
    /// ordinals are selected from this list; CPU remains available whenever it
    /// appears in `available_devices`, even without a concrete descriptor.
    pub devices: Vec<DeviceDescriptor>,
    pub host_to_device_bytes_per_sec: u64,
    pub device_to_host_bytes_per_sec: u64,
    pub kernel_launch_seconds: f64,
    pub synchronization_seconds: f64,
    pub host_to_device_latency_seconds: f64,
    pub device_to_host_latency_seconds: f64,
    pub device_to_device_latency_seconds: f64,
    pub allocation_latency_seconds: f64,
    pub host_memory_bytes_per_sec: u64,
    pub device_memory_bytes_per_sec: u64,
    pub device_to_device_bytes_per_sec: u64,
    pub cpu_compute_ops_per_sec: f64,
    pub device_compute_ops_per_sec: f64,
    /// Conservative byte estimate used when shape or dtype is unknown.
    pub unknown_value_bytes: u64,
    pub preferred_sink_device: Option<DeviceClass>,
}

impl Default for MachineProfile {
    fn default() -> Self {
        Self {
            cpu_threads: 1,
            available_devices: vec![DeviceClass::Cpu],
            devices: Vec::new(),
            host_to_device_bytes_per_sec: 12_000_000_000,
            device_to_host_bytes_per_sec: 12_000_000_000,
            kernel_launch_seconds: 0.000_01,
            synchronization_seconds: 0.000_01,
            host_to_device_latency_seconds: 0.000_02,
            device_to_host_latency_seconds: 0.000_02,
            device_to_device_latency_seconds: 0.000_01,
            allocation_latency_seconds: 0.000_001,
            host_memory_bytes_per_sec: 50_000_000_000,
            device_memory_bytes_per_sec: 700_000_000_000,
            device_to_device_bytes_per_sec: 100_000_000_000,
            cpu_compute_ops_per_sec: 100_000_000_000.0,
            device_compute_ops_per_sec: 10_000_000_000_000.0,
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
    /// Concrete device selected when the machine profile provides inventory.
    /// `None` means only a device class was available to the planner.
    pub location: Option<DeviceLocation>,
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
    pub transfer_boundaries: Vec<TransferBoundary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// Explicit placement-time transfer requirement. Physical planning lowers
/// this record into an executable transfer node.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferBoundary {
    pub before: NodeId,
    /// Producer of this edge; present in graph placement.
    pub input: Option<NodeId>,
    pub from: DeviceClass,
    pub to: DeviceClass,
    pub kind: TransferKind,
    pub bytes: u64,
    pub estimated_seconds: f64,
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
                "  %{} {:?} kernel={} device={:?} location={:?} parallelism={} cost={:.6}s host={}B(r{} w{}) device={}B(r{} w{}) transfer={}B({:.6}s) temp={}B alloc={} launch={} sync={} compute={:.6}s alignment={} contiguity={:?} fusion={:?} candidates={:?} shape={:?} dtype={:?} layout={:?}\n",
                selected.node.index(), node.kind(), selected.kernel, selected.device, selected.location,
                selected.parallelism, selected.cost.total_score, selected.cost.host_bytes,
                selected.cost.host_bytes_read, selected.cost.host_bytes_written,
                selected.cost.device_bytes, selected.cost.device_bytes_read,
                selected.cost.device_bytes_written, selected.cost.transfer_bytes,
                selected.cost.transfer_score,
                selected.cost.temporary_bytes,
                selected.cost.allocation_count, selected.cost.launch_count,
                selected.cost.synchronization_count, selected.cost.compute_score,
                selected.alignment_bytes, selected.contiguity, selected.fusion_tags, selected.alternatives,
                props.and_then(|p| p.shape.as_ref()), props.and_then(|p| p.dtype.as_ref()),
                props.and_then(|p| p.axis_order.as_ref()),
            ));
        }
        for boundary in &self.transfer_boundaries {
            out.push_str(&format!(
                "  transfer before %{}: {:?} {:?} -> {:?} {}B estimated={:.6}s\n",
                boundary.before.index(),
                boundary.kind,
                boundary.from,
                boundary.to,
                boundary.bytes,
                boundary.estimated_seconds,
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
    #[error("placement currently supports linear pipelines; node %{node} has {inputs} inputs")]
    NonLinearInputs { node: usize, inputs: usize },
    #[error(
        "placement currently supports linear pipelines; reachable node %{node} has multiple consumers"
    )]
    NonLinearConsumers { node: usize },
    #[error("sink requires {device:?}, but no compatible registered kernel can produce it")]
    SinkDeviceUnavailable { device: DeviceClass },
}

/// Select registered implementations. Linear graphs use exact dynamic
/// programming; DAGs use a deterministic topological heuristic that costs all
/// incoming edges. Kernel traffic, launch and transfer costs guide selection.
pub fn place(
    plan: &LogicalPlan,
    annotations: &PropertyAnnotations,
    capabilities: &KernelCapabilities,
    machine: &MachineProfile,
) -> Result<PlacementPlan, PlacementError> {
    plan.validate()?;
    let ids = match linear_pipeline_order(plan) {
        Ok(ids) => ids,
        Err(PlacementError::NonLinearInputs { .. } | PlacementError::NonLinearConsumers { .. }) => {
            return place_dag(plan, annotations, capabilities, machine);
        }
        Err(error) => return Err(error),
    };
    let mut candidate_rows = Vec::with_capacity(ids.len());
    let mut local_cost_rows = Vec::with_capacity(ids.len());

    for (position, id) in ids.iter().copied().enumerate() {
        let props = annotations.get(id);
        let input = input_properties(plan, id, annotations);
        let output = props;
        let available = capabilities.for_node(plan, id)?;
        let has_registered = !available.is_empty();
        let mut compatible = available
            .into_iter()
            .filter(|kernel| {
                device_supports_kernel(
                    machine,
                    kernel,
                    required_device_bytes(kernel, input, output, machine),
                )
            })
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
        // A sink candidate describes the device required by its consumed value.
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
        let local_costs = compatible
            .iter()
            .map(|kernel| estimate_cost(kernel, input, output, machine))
            .collect::<Vec<_>>();
        candidate_rows.push(compatible);
        local_cost_rows.push(local_costs);
    }

    // dp[position][candidate] stores the minimum cost ending at that kernel.
    // Backpointers recover the globally optimal device/kernel path.
    let mut dp: Vec<Vec<f64>> = Vec::with_capacity(ids.len());
    let mut back: Vec<Vec<Option<usize>>> = Vec::with_capacity(ids.len());
    for position in 0..ids.len() {
        let mut scores = vec![f64::INFINITY; candidate_rows[position].len()];
        let mut predecessors = vec![None; candidate_rows[position].len()];
        for candidate_index in 0..candidate_rows[position].len() {
            let kernel = candidate_rows[position][candidate_index];
            let local_score = local_cost_rows[position][candidate_index].total_score;
            if position == 0 {
                scores[candidate_index] = local_score;
                continue;
            }
            let edge_bytes = input_properties(plan, ids[position], annotations)
                .and_then(byte_size)
                .unwrap_or(machine.unknown_value_bytes);
            for previous_index in 0..candidate_rows[position - 1].len() {
                let previous = candidate_rows[position - 1][previous_index];
                let transfer = transfer_cost(
                    ids[position],
                    &previous.device,
                    &kernel.device,
                    edge_bytes,
                    machine,
                );
                let score =
                    dp[position - 1][previous_index] + transfer.estimated_seconds + local_score;
                if score.total_cmp(&scores[candidate_index]).is_lt() {
                    scores[candidate_index] = score;
                    predecessors[candidate_index] = Some(previous_index);
                }
            }
        }
        dp.push(scores);
        back.push(predecessors);
    }

    let last = ids.len() - 1;
    let mut selected_indices = vec![0; ids.len()];
    selected_indices[last] = (0..dp[last].len())
        .min_by(|lhs, rhs| dp[last][*lhs].total_cmp(&dp[last][*rhs]))
        .expect("each pipeline node has a compatible candidate");
    for position in (1..ids.len()).rev() {
        selected_indices[position - 1] = back[position][selected_indices[position]]
            .expect("every non-source candidate has a predecessor");
    }

    let mut output = PlacementPlan::default();
    for position in 0..ids.len() {
        let id = ids[position];
        let selected = candidate_rows[position][selected_indices[position]];
        let parallelism = if selected.parallel {
            machine.cpu_threads.max(1)
        } else {
            1
        };
        let mut cost = local_cost_rows[position][selected_indices[position]].clone();
        if position > 0 {
            let previous = candidate_rows[position - 1][selected_indices[position - 1]];
            let edge_bytes = input_properties(plan, id, annotations)
                .and_then(byte_size)
                .unwrap_or(machine.unknown_value_bytes);
            let mut transfer =
                transfer_cost(id, &previous.device, &selected.device, edge_bytes, machine);
            transfer.input = Some(ids[position - 1]);
            if previous.device != selected.device {
                cost.transfer_bytes = cost.transfer_bytes.saturating_add(transfer.bytes);
                cost.transfer_score += transfer.estimated_seconds;
                cost.total_score += transfer.estimated_seconds;
                output.transfer_boundaries.push(transfer);
            }
        }
        let alternatives = candidate_rows[position]
            .iter()
            .enumerate()
            .map(|(candidate_index, candidate)| {
                let prefix = if position == 0 {
                    0.0
                } else {
                    let edge_bytes = input_properties(plan, id, annotations)
                        .and_then(byte_size)
                        .unwrap_or(machine.unknown_value_bytes);
                    candidate_rows[position - 1]
                        .iter()
                        .enumerate()
                        .map(|(previous_index, previous)| {
                            dp[position - 1][previous_index]
                                + transfer_cost(
                                    id,
                                    &previous.device,
                                    &candidate.device,
                                    edge_bytes,
                                    machine,
                                )
                                .estimated_seconds
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                format!(
                    "{}@{:?}={:.6}",
                    candidate.name,
                    candidate.device,
                    prefix + local_cost_rows[position][candidate_index].total_score
                )
            })
            .collect();
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
            location: selected_device_location(
                machine,
                selected,
                required_device_bytes(
                    selected,
                    input_properties(plan, id, annotations),
                    annotations.get(id),
                    machine,
                ),
            ),
        });
    }
    Ok(output)
}

/// Deterministic graph placement. Choose a compatible kernel in topological
/// order, accounting for every incoming transfer. The linear path retains its
/// exact global dynamic program; this graph heuristic is not globally optimal.
fn place_dag(
    plan: &LogicalPlan,
    annotations: &PropertyAnnotations,
    capabilities: &KernelCapabilities,
    machine: &MachineProfile,
) -> Result<PlacementPlan, PlacementError> {
    let mut result = PlacementPlan::default();
    let mut selected_devices = HashMap::<NodeId, DeviceClass>::new();
    for id in plan.topological_order()? {
        let node = plan.node(id)?;
        let output = annotations.get(id);
        let input = input_properties(plan, id, annotations);
        let mut choices = Vec::new();
        for kernel in capabilities.for_node(plan, id)? {
            if !device_supports_kernel(
                machine,
                kernel,
                required_graph_device_bytes(plan, id, kernel, annotations, machine),
            ) || !properties_match(&kernel.requirements, plan, id, annotations)
                || (kernel.contiguity != Contiguity::Unknown
                    && output.is_none_or(|p| p.contiguity != Some(kernel.contiguity)))
                || (node.kind() == NodeKind::Sink
                    && machine
                        .preferred_sink_device
                        .as_ref()
                        .is_some_and(|device| device != &kernel.device))
            {
                continue;
            }
            let mut cost = estimate_cost(kernel, input, output, machine);
            // Cost hints apply to total incoming traffic rather than just port 0.
            for extra in node.inputs().iter().skip(1) {
                let extra_cost = estimate_cost(kernel, annotations.get(extra), output, machine);
                cost.host_bytes_read = cost
                    .host_bytes_read
                    .saturating_add(extra_cost.host_bytes_read);
                cost.device_bytes_read = cost
                    .device_bytes_read
                    .saturating_add(extra_cost.device_bytes_read);
                cost.host_bytes = cost.host_bytes.saturating_add(extra_cost.host_bytes_read);
                cost.device_bytes = cost
                    .device_bytes
                    .saturating_add(extra_cost.device_bytes_read);
                let extra_score = extra_cost.host_bytes_read as f64
                    / machine.host_memory_bytes_per_sec.max(1) as f64
                    + extra_cost.device_bytes_read as f64
                        / machine.device_memory_bytes_per_sec.max(1) as f64;
                cost.compute_score += extra_score;
                cost.total_score += extra_score;
            }
            let mut transfers = Vec::new();
            for producer in node.inputs().iter() {
                let from = &selected_devices[&producer];
                if from != &kernel.device {
                    let bytes = annotations
                        .get(producer)
                        .and_then(byte_size)
                        .unwrap_or(machine.unknown_value_bytes);
                    let mut transfer = transfer_cost(id, from, &kernel.device, bytes, machine);
                    transfer.input = Some(producer);
                    cost.transfer_bytes = cost.transfer_bytes.saturating_add(bytes);
                    cost.transfer_score += transfer.estimated_seconds;
                    cost.total_score += transfer.estimated_seconds;
                    transfers.push(transfer);
                }
            }
            choices.push((kernel, cost, transfers));
        }
        let Some(selected_index) = (0..choices.len()).min_by(|a, b| {
            choices[*a]
                .1
                .total_score
                .total_cmp(&choices[*b].1.total_score)
        }) else {
            if node.kind() == NodeKind::Sink
                && let Some(device) = &machine.preferred_sink_device
            {
                return Err(PlacementError::SinkDeviceUnavailable {
                    device: device.clone(),
                });
            }
            return Err(PlacementError::NoKernel {
                node: id.index(),
                reason: "no compatible registered kernel for graph node".to_owned(),
            });
        };
        let alternatives = choices
            .iter()
            .map(|(k, c, _)| format!("{}@{:?}={:.6}", k.name, k.device, c.total_score))
            .collect();
        let (kernel, cost, transfers) = choices.swap_remove(selected_index);
        selected_devices.insert(id, kernel.device.clone());
        result.transfer_boundaries.extend(transfers);
        result.candidates.push(PlacementCandidate {
            node: id,
            kernel: kernel.name.clone(),
            device: kernel.device.clone(),
            class: kernel.class,
            parallelism: if kernel.parallel {
                machine.cpu_threads.max(1)
            } else {
                1
            },
            cost,
            alternatives,
            fusion_tags: kernel.fusion_tags.clone(),
            alignment_bytes: kernel.alignment_bytes,
            contiguity: kernel.contiguity,
            location: selected_device_location(
                machine,
                kernel,
                required_graph_device_bytes(plan, id, kernel, annotations, machine),
            ),
        });
    }
    Ok(result)
}

fn required_graph_device_bytes(
    plan: &LogicalPlan,
    id: NodeId,
    kernel: &KernelCapability,
    annotations: &PropertyAnnotations,
    machine: &MachineProfile,
) -> u64 {
    if kernel.device == DeviceClass::Cpu {
        return 0;
    }
    let input_bytes =
        plan.node(id)
            .expect("validated node")
            .inputs()
            .iter()
            .fold(0u64, |bytes, input| {
                bytes.saturating_add(
                    annotations
                        .get(input)
                        .and_then(byte_size)
                        .unwrap_or(machine.unknown_value_bytes),
                )
            });
    let output_bytes = annotations
        .get(id)
        .and_then(byte_size)
        .unwrap_or(machine.unknown_value_bytes);
    match kernel.class {
        KernelClass::Source => output_bytes.saturating_add(kernel.temporary_bytes as u64),
        KernelClass::Sink => input_bytes,
        _ => input_bytes
            .saturating_add(output_bytes)
            .saturating_add(kernel.temporary_bytes as u64),
    }
}

fn linear_pipeline_order(plan: &LogicalPlan) -> Result<Vec<NodeId>, PlacementError> {
    let reachable = plan.preorder()?;
    let reachable_set = reachable.iter().copied().collect::<HashSet<_>>();
    for id in &reachable {
        let node = plan.node(*id)?;
        if node.inputs().len() > 1 {
            return Err(PlacementError::NonLinearInputs {
                node: id.index(),
                inputs: node.inputs().len(),
            });
        }
        let consumers = plan
            .children(*id)?
            .into_iter()
            .filter(|consumer| reachable_set.contains(consumer))
            .count();
        if consumers > 1 {
            return Err(PlacementError::NonLinearConsumers { node: id.index() });
        }
    }

    let mut reverse = Vec::with_capacity(reachable.len());
    let mut current = plan.root()?;
    loop {
        reverse.push(current);
        let node = plan.node(current)?;
        match node.inputs().get(0) {
            Some(input) => current = input,
            None => break,
        }
    }
    reverse.reverse();
    if reverse.len() != reachable.len() {
        let node = reachable
            .iter()
            .find(|id| !reverse.contains(id))
            .copied()
            .unwrap_or(plan.root()?);
        return Err(PlacementError::NonLinearConsumers { node: node.index() });
    }
    Ok(reverse)
}

fn input_properties<'a>(
    plan: &LogicalPlan,
    id: NodeId,
    annotations: &'a PropertyAnnotations,
) -> Option<&'a ValueProperties> {
    let input = plan.node(id).ok()?.inputs().get(0)?;
    annotations.get(input)
}

fn device_supports_kernel(
    machine: &MachineProfile,
    kernel: &KernelCapability,
    required_memory_bytes: u64,
) -> bool {
    if kernel.device == DeviceClass::Cpu {
        return true;
    }
    if machine.devices.is_empty() {
        machine.available_devices.contains(&kernel.device)
    } else {
        machine.devices.iter().any(|device| {
            device.class == kernel.device
                && (device.supported_kernels.is_empty()
                    || device.supported_kernels.contains(&kernel.name))
                && device
                    .memory_bytes
                    .is_none_or(|memory| memory >= required_memory_bytes)
        })
    }
}

fn required_device_bytes(
    kernel: &KernelCapability,
    input: Option<&ValueProperties>,
    output: Option<&ValueProperties>,
    machine: &MachineProfile,
) -> u64 {
    if kernel.device == DeviceClass::Cpu {
        return 0;
    }
    let input_bytes = input
        .map(|properties| byte_size(properties).unwrap_or(machine.unknown_value_bytes))
        .unwrap_or(0);
    let output_bytes = output
        .map(|properties| byte_size(properties).unwrap_or(machine.unknown_value_bytes))
        .unwrap_or(0);
    match kernel.class {
        KernelClass::Source => output_bytes.saturating_add(kernel.temporary_bytes as u64),
        KernelClass::Sink => input_bytes,
        _ => input_bytes
            .saturating_add(output_bytes)
            .saturating_add(kernel.temporary_bytes as u64),
    }
}

fn selected_device_location(
    machine: &MachineProfile,
    kernel: &KernelCapability,
    required_memory_bytes: u64,
) -> Option<DeviceLocation> {
    if kernel.device == DeviceClass::Cpu {
        return Some(DeviceLocation::Cpu);
    }
    machine
        .devices
        .iter()
        .find(|device| {
            device.class == kernel.device
                && (device.supported_kernels.is_empty()
                    || device.supported_kernels.contains(&kernel.name))
                && device
                    .memory_bytes
                    .is_none_or(|memory| memory >= required_memory_bytes)
        })
        .map(DeviceDescriptor::location)
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
    (!has_input_requirements
        || (!node.inputs().is_empty()
            && node
                .inputs()
                .iter()
                .all(|id| matches_input(annotations.get(id)))))
        && matches_output
}

fn estimate_cost(
    kernel: &KernelCapability,
    input: Option<&ValueProperties>,
    output: Option<&ValueProperties>,
    machine: &MachineProfile,
) -> CostEstimate {
    let temporary_bytes = kernel.temporary_bytes as u64;
    let input_bytes = input
        .map(|properties| byte_size(properties).unwrap_or(machine.unknown_value_bytes))
        .unwrap_or(0);
    let output_bytes = output
        .map(|properties| byte_size(properties).unwrap_or(machine.unknown_value_bytes))
        .unwrap_or(0);
    let device = !matches!(kernel.device, DeviceClass::Cpu);
    let (read_bytes, written_bytes) = match kernel.class {
        KernelClass::Source => (output_bytes, 0),
        KernelClass::Sink => (input_bytes, 0),
        _ => (
            input_bytes.saturating_mul(kernel.cost_hint.input_read_passes as u64),
            output_bytes.saturating_mul(kernel.cost_hint.output_write_passes as u64),
        ),
    };
    let memory_traffic = read_bytes
        .saturating_add(written_bytes)
        .saturating_add(temporary_bytes.saturating_mul(2));
    let memory_bandwidth = if device {
        machine.device_memory_bytes_per_sec
    } else {
        machine.host_memory_bytes_per_sec
    }
    .max(1);
    // This first model estimates the memory-bound portion from explicit input,
    // output, and scratch traffic. Kernel-specific compute calibration can be
    // added without changing the path optimizer.
    let memory_score = memory_traffic as f64 / memory_bandwidth as f64;
    let element_count = output
        .and_then(element_count)
        .or_else(|| input.and_then(element_count))
        .unwrap_or(machine.unknown_value_bytes);
    let compute_ops = if matches!(kernel.class, KernelClass::Source | KernelClass::Sink) {
        0
    } else {
        element_count.saturating_mul(kernel.cost_hint.compute_ops_per_element as u64)
    };
    let compute_throughput = if device {
        machine.device_compute_ops_per_sec
    } else {
        machine.cpu_compute_ops_per_sec
    }
    .max(1.0);
    let compute_score = memory_score + compute_ops as f64 / compute_throughput;
    let allocation_count = u32::from(!kernel.in_place);
    let launch_count = u32::from(
        device && kernel.class != KernelClass::Source && kernel.class != KernelClass::Sink,
    );
    let synchronization_count = launch_count;
    let total_score = compute_score
        + allocation_count as f64 * machine.allocation_latency_seconds
        + launch_count as f64 * machine.kernel_launch_seconds
        + synchronization_count as f64 * machine.synchronization_seconds;
    CostEstimate {
        host_bytes: if device { 0 } else { memory_traffic },
        device_bytes: if device { memory_traffic } else { 0 },
        host_bytes_read: if device { 0 } else { read_bytes },
        host_bytes_written: if device { 0 } else { written_bytes },
        device_bytes_read: if device { read_bytes } else { 0 },
        device_bytes_written: if device { written_bytes } else { 0 },
        transfer_bytes: 0,
        temporary_bytes,
        allocation_count,
        launch_count,
        synchronization_count,
        compute_score,
        transfer_score: 0.0,
        total_score,
    }
}

fn transfer_cost(
    before: NodeId,
    from: &DeviceClass,
    to: &DeviceClass,
    bytes: u64,
    machine: &MachineProfile,
) -> TransferBoundary {
    if from == to {
        return TransferBoundary {
            before,
            input: None,
            from: from.clone(),
            to: to.clone(),
            kind: TransferKind::DeviceToDevice,
            bytes: 0,
            estimated_seconds: 0.0,
        };
    }
    let (kind, latency, bandwidth) = match (from, to) {
        (DeviceClass::Cpu, _) => (
            TransferKind::HostToDevice,
            machine.host_to_device_latency_seconds,
            machine.host_to_device_bytes_per_sec,
        ),
        (_, DeviceClass::Cpu) => (
            TransferKind::DeviceToHost,
            machine.device_to_host_latency_seconds,
            machine.device_to_host_bytes_per_sec,
        ),
        _ => (
            TransferKind::DeviceToDevice,
            machine.device_to_device_latency_seconds,
            machine.device_to_device_bytes_per_sec,
        ),
    };
    TransferBoundary {
        before,
        input: None,
        from: from.clone(),
        to: to.clone(),
        kind,
        bytes,
        estimated_seconds: latency + bytes as f64 / bandwidth.max(1) as f64,
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

fn element_count(properties: &ValueProperties) -> Option<u64> {
    properties
        .shape
        .as_ref()?
        .dims()
        .iter()
        .try_fold(1u64, |size, dim| match dim {
            ShapeDim::Known(value) => size.checked_mul(*value as u64),
            ShapeDim::Dynamic => None,
        })
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
            cost_hint: KernelCostHint::default(),
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
    #[test]
    fn dag_placement_costs_each_incoming_edge_and_checks_all_ports() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(
            NodeKind::Source,
            [],
            Some(std::sync::Arc::new(Payload)),
        ));
        let join = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source, source],
            Some(std::sync::Arc::new(Payload)),
        ));
        plan.set_root(join).unwrap();
        let mut props = PropertyAnnotations::default();
        let properties = ValueProperties {
            dtype: Some(DataType::U8),
            shape: Some(crate::ValueShape(vec![ShapeDim::Known(8)])),
            ..ValueProperties::default()
        };
        props.insert(source, properties.clone());
        props.insert(join, properties);
        let mut caps = KernelCapabilities::default();
        for (kind, class, device) in [
            (NodeKind::Source, KernelClass::Source, DeviceClass::Cpu),
            (NodeKind::Op, KernelClass::Batch, DeviceClass::Cuda),
        ] {
            caps.register(KernelCapability {
                name: format!("demo-{class:?}"),
                operator: "demo::Op".into(),
                node_kind: kind,
                device,
                class,
                requirements: if kind == NodeKind::Op {
                    KernelRequirements {
                        input_dtype: Some(DataType::U8),
                        ..KernelRequirements::default()
                    }
                } else {
                    KernelRequirements::default()
                },
                fusion_tags: Vec::new(),
                alignment_bytes: 1,
                contiguity: Contiguity::Unknown,
                temporary_bytes: 0,
                cost_hint: KernelCostHint::default(),
                in_place: false,
                parallel: false,
            });
        }
        let machine = MachineProfile {
            available_devices: vec![DeviceClass::Cpu, DeviceClass::Cuda],
            ..MachineProfile::default()
        };
        let selected = place(&plan, &props, &caps, &machine).unwrap();
        assert_eq!(selected.candidates.len(), 2);
        assert_eq!(selected.transfer_boundaries.len(), 2);
        assert!(
            selected
                .transfer_boundaries
                .iter()
                .all(|edge| edge.input == Some(source) && edge.before == join && edge.bytes == 8)
        );
        assert_eq!(selected.candidates[1].cost.device_bytes_read, 16);
        assert_eq!(selected.candidates[1].cost.device_bytes, 24);
        assert_eq!(selected.candidates[1].cost.transfer_bytes, 16);
        let other = plan.add_node(LogicalNode::new(
            NodeKind::Source,
            [],
            Some(std::sync::Arc::new(Payload)),
        ));
        props.insert(
            other,
            ValueProperties {
                dtype: Some(DataType::F32),
                ..ValueProperties::default()
            },
        );
        plan.replace_node(
            join,
            LogicalNode::new(
                NodeKind::Op,
                [source, other],
                Some(std::sync::Arc::new(Payload)),
            ),
        )
        .unwrap();
        assert!(
            matches!(place(&plan, &props, &caps, &machine), Err(PlacementError::NoKernel { node, .. }) if node == join.index())
        );
    }
}
