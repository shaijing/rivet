//! Buffer lifetime and reuse planning for a validated physical graph.
//!
//! This module produces an allocation plan; device-specific pools and actual
//! allocation remain owned by the execution backend.

use std::collections::HashMap;

use rivet_core::DType;

use crate::physical::{ExecutionLane, MorselResidency, PhysNodeId, PhysicalGraph};

/// Memory constraints declared by a physical kernel capability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelMemoryRequirements {
    pub input_contiguous: bool,
    pub output_contiguous: bool,
    pub output_alignment: usize,
    pub temporary_bytes: usize,
    pub in_place: bool,
}

/// Buffers may share a pool slot only when all layout and residency
/// requirements represented here match.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferReuseClass {
    pub residency: MorselResidency,
    pub dtype: Option<DType>,
    pub shape: Option<Vec<usize>>,
    pub alignment: usize,
    pub contiguous: bool,
}

impl BufferReuseClass {
    pub fn for_lane(lane: &ExecutionLane, requirements: KernelMemoryRequirements) -> Self {
        let residency = match lane {
            ExecutionLane::Device { class, ordinal } => MorselResidency::Device {
                class: class.clone(),
                ordinal: *ordinal,
            },
            ExecutionLane::Io | ExecutionLane::Cpu | ExecutionLane::Transfer => {
                MorselResidency::Host
            }
        };
        Self {
            residency,
            dtype: None,
            shape: None,
            alignment: requirements.output_alignment.max(1),
            contiguous: requirements.output_contiguous,
        }
    }
}

/// One node's expected output allocation. Unknown sizes are tracked for
/// lifetime analysis but are not assigned to a reusable capacity slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRequest {
    pub size_bytes: Option<usize>,
    pub reuse_class: BufferReuseClass,
    pub requirements: KernelMemoryRequirements,
}

impl MemoryRequest {
    pub fn new(
        size_bytes: Option<usize>,
        reuse_class: BufferReuseClass,
        requirements: KernelMemoryRequirements,
    ) -> Self {
        Self {
            size_bytes,
            reuse_class,
            requirements,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedAllocation {
    pub node: PhysNodeId,
    pub first_use: usize,
    pub last_use: usize,
    pub size_bytes: Option<usize>,
    pub temporary_bytes: usize,
    pub slot: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSlot {
    pub id: usize,
    pub reuse_class: BufferReuseClass,
    pub capacity_bytes: usize,
    pub last_use: usize,
    pub allocations: Vec<PhysNodeId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryPlan {
    pub allocations: Vec<PlannedAllocation>,
    pub slots: Vec<BufferSlot>,
    /// Sum of capacities the backend would need for the reusable pool.
    pub pool_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryPlanner;

impl MemoryPlanner {
    /// Build requests from output-size and kernel-memory metadata already
    /// attached during physical lowering.
    pub fn plan_physical(
        graph: &mut PhysicalGraph,
    ) -> Result<MemoryPlan, crate::physical::PhysicalGraphError> {
        let requests = graph
            .nodes()
            .iter()
            .map(|node| {
                let requirements = node.memory_requirements;
                (
                    node.id,
                    MemoryRequest::new(
                        node.estimated_output_bytes,
                        BufferReuseClass::for_lane(&node.lane, requirements),
                        requirements,
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        Self::plan(graph, &requests)
    }

    /// Build a conservative interval-coloring plan. A slot is reused only
    /// after the previous value's final consumer and only for an identical
    /// reuse class with enough capacity.
    pub fn plan(
        graph: &mut PhysicalGraph,
        requests: &HashMap<PhysNodeId, MemoryRequest>,
    ) -> Result<MemoryPlan, crate::physical::PhysicalGraphError> {
        graph.validate()?;
        let order = graph.execution_order()?;
        let mut positions = vec![0usize; graph.nodes().len()];
        for (position, id) in order.iter().copied().enumerate() {
            positions[id.index()] = position;
        }

        let mut last_use = positions.clone();
        for node in graph.nodes() {
            for input in &node.inputs {
                last_use[input.index()] = last_use[input.index()].max(positions[node.id.index()]);
            }
        }
        // The root value remains live until the caller consumes it.
        let root = graph.root()?;
        last_use[root.index()] = order.len();

        let mut plan = MemoryPlan::default();
        for id in order {
            let Some(request) = requests.get(&id) else {
                continue;
            };
            let start = positions[id.index()];
            let end = last_use[id.index()];
            let slot = request.size_bytes.and_then(|size| {
                let in_place_slot = if request.requirements.in_place {
                    graph.node(id).ok().and_then(|node| {
                        let [input] = node.inputs.as_slice() else {
                            return None;
                        };
                        let input_request = requests.get(input)?;
                        let input_allocation = plan
                            .allocations
                            .iter()
                            .find(|allocation| allocation.node == *input)?;
                        let slot_id = input_allocation.slot?;
                        (graph.node(*input).ok()?.last_use_count == 1
                            && input_allocation.last_use == start
                            && input_request.reuse_class == request.reuse_class
                            && plan.slots[slot_id].capacity_bytes >= size)
                            .then_some(slot_id)
                    })
                } else {
                    None
                };
                if let Some(slot_id) = in_place_slot {
                    let slot = &mut plan.slots[slot_id];
                    slot.last_use = end;
                    slot.allocations.push(id);
                    return Some(slot_id);
                }
                let candidate = plan.slots.iter().position(|slot| {
                    slot.last_use < start
                        && slot.reuse_class == request.reuse_class
                        && slot.capacity_bytes >= size
                });
                if let Some(slot_id) = candidate {
                    let slot = &mut plan.slots[slot_id];
                    slot.last_use = end;
                    slot.allocations.push(id);
                    Some(slot_id)
                } else {
                    let slot_id = plan.slots.len();
                    plan.slots.push(BufferSlot {
                        id: slot_id,
                        reuse_class: request.reuse_class.clone(),
                        capacity_bytes: size,
                        last_use: end,
                        allocations: vec![id],
                    });
                    plan.pool_bytes = plan.pool_bytes.saturating_add(size);
                    Some(slot_id)
                }
            });
            plan.allocations.push(PlannedAllocation {
                node: id,
                first_use: start,
                last_use: end,
                size_bytes: request.size_bytes,
                temporary_bytes: request.requirements.temporary_bytes,
                slot,
            });
        }
        Ok(plan)
    }
}
