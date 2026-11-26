//! Memory limits and observation scope (CPU).
//!
//! * Queue bytes are the sum of producer-reported payload weights currently
//!   enqueued on one edge. Dequeue releases that queue's weight immediately;
//!   executing values, reorder buffers and caller outputs are outside it.
//! * Value logical bytes are a domain-reported output size, checked AFTER a
//!   kernel returns. They are neither backing capacity nor a memory reservation.
//! * Observed graph-live backing bytes count distinct tensor storage base
//!   addresses held at a node boundary of ONE graph invocation. A view charges
//!   its entire backing storage once. This is an opt-in, post-allocation check,
//!   not a shared executor budget or a measure of the peak inside a kernel.
//! * DAG managed admission independently reserves declared allocation
//!   capacity for queued, executing and reordered morsels across workers.
//!
//! Observation includes input backing storage still held by the graph, even
//! when allocated externally. It excludes scratch and temporary allocations
//! inside kernels/decoders, object/Vec metadata, cache entries not referenced by
//! the graph, completed worker results/reorder buffers outside this invocation,
//! caller-held outputs after return, allocator rounding and external libraries.
//! Known backing bytes use Tensor::storage_bytes (full addressable storage), not
//! logical view bytes or RSS. Opaque Encoded/Decoded values and legacy byte-only
//! morsels have unknown backing capacity; they make a snapshot incomplete,
//! never an exact zero-byte value. A limit on observed backing fails closed on
//! incomplete snapshots. Even complete snapshots are not process RSS limits.

use std::collections::{HashMap, HashSet};

use crate::physical::{ExecutionValue, Morsel};

/// Separate limits for one physical graph invocation. Default execution keeps
/// the existing fast path; accounting is collected only for profiling or an
/// explicit observed-live check. No cross-worker managed budget is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphMemoryLimits {
    /// Producer-reported logical output weight; legacy zero may be unreported.
    /// This caps the reported size, not an unknown backing allocation.
    pub max_value_logical_bytes: usize,
    /// Opt-in boundary observation limit; incomplete footprints fail closed.
    pub max_observed_live_backing_bytes: Option<usize>,
}
impl Default for GraphMemoryLimits {
    fn default() -> Self {
        Self {
            max_value_logical_bytes: usize::MAX,
            max_observed_live_backing_bytes: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveMemorySnapshot {
    /// Sum of logical tensor views; aliases may contribute more than once.
    pub logical_tensor_bytes: usize,
    /// Distinct full tensor backing storages; aliases contribute once.
    pub known_backing_bytes: usize,
    pub unique_backing_allocations: usize,
    /// Distinct opaque envelopes plus legacy payloads with no backing metadata.
    /// A shared opaque envelope still has unknown capacity, even if counted once.
    pub unknown_values: usize,
    pub overflowed: bool,
}
impl LiveMemorySnapshot {
    pub fn is_complete(&self) -> bool {
        self.unknown_values == 0 && !self.overflowed
    }

    /// Compare observed storage only; this does not reserve or cap allocations.
    pub fn check_observed_limit(&self, limit: usize) -> Result<(), MemoryAccountingError> {
        if self.overflowed {
            return Err(MemoryAccountingError::Overflow);
        }
        if self.known_backing_bytes > limit {
            return Err(MemoryAccountingError::ObservedLiveExceeded {
                bytes: self.known_backing_bytes,
                limit,
            });
        }
        if self.unknown_values > 0 {
            return Err(MemoryAccountingError::Incomplete {
                unknown_values: self.unknown_values,
            });
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MemoryAccountingError {
    #[error(
        "observed graph-live backing bytes exceed limit after allocation: {bytes} > {limit}; one invocation at node boundaries, not a managed executor budget"
    )]
    ObservedLiveExceeded { bytes: usize, limit: usize },
    #[error(
        "cannot enforce observed graph-live backing limit: {unknown_values} values have unknown backing capacity"
    )]
    Incomplete { unknown_values: usize },
    #[error("graph memory observation overflowed; exact byte accounting is unavailable")]
    Overflow,
}

/// Snapshot allocation identity only while the supplied values are alive.
/// Addresses are never accumulated across lifetimes, where an allocator could
/// reuse an address. Empty tensor storage has no backing allocation to charge.
pub fn observe_live_morsels<'a>(
    morsels: impl IntoIterator<Item = &'a Morsel>,
) -> LiveMemorySnapshot {
    let mut snapshot = LiveMemorySnapshot::default();
    let mut allocations = HashMap::<usize, usize>::new();
    let mut opaque = HashSet::new();
    for morsel in morsels {
        if morsel.values.is_empty() && morsel.bytes > 0 {
            snapshot.unknown_values = snapshot.unknown_values.saturating_add(1);
        }
        let mut visit = |tensor: &rivet_core::Tensor| {
            snapshot.logical_tensor_bytes = add_bytes(
                snapshot.logical_tensor_bytes,
                tensor.logical_bytes(),
                &mut snapshot.overflowed,
            );
            let bytes = tensor.storage_bytes();
            if bytes != 0 {
                let id = tensor.storage_base_ptr() as usize;
                allocations
                    .entry(id)
                    .and_modify(|capacity| *capacity = (*capacity).max(bytes))
                    .or_insert(bytes);
            }
        };
        for value in &morsel.values {
            match value {
                ExecutionValue::Tensor(tensor) => visit(tensor),
                ExecutionValue::DecodedWithTensors(value, tensors) => {
                    tensors(value.as_ref(), &mut visit)
                }
                ExecutionValue::Encoded(value) | ExecutionValue::Decoded(value) => {
                    opaque.insert(std::sync::Arc::as_ptr(value) as *const () as usize);
                }
            }
        }
    }
    snapshot.unknown_values = snapshot.unknown_values.saturating_add(opaque.len());
    snapshot.unique_backing_allocations = allocations.len();
    for bytes in allocations.into_values() {
        snapshot.known_backing_bytes = add_bytes(
            snapshot.known_backing_bytes,
            bytes,
            &mut snapshot.overflowed,
        );
    }
    snapshot
}
fn add_bytes(a: usize, b: usize, overflowed: &mut bool) -> usize {
    match a.checked_add(b) {
        Some(sum) => sum,
        None => {
            *overflowed = true;
            usize::MAX
        }
    }
}

/// Maxima across boundary observations of separate graph invocations, NOT a
/// sum of simultaneous worker memory. Unknown footprints remain explicit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphMemoryProfile {
    pub boundary_observations: u64,
    pub peak_known_backing_bytes_per_invocation: u64,
    pub peak_unknown_values_per_invocation: u64,
    pub incomplete_observations: u64,
    pub value_logical_limit_failures: u64,
    pub observed_live_limit_failures: u64,
}

#[cfg(test)]
#[path = "accounting_tests.rs"]
mod tests;

impl GraphMemoryProfile {
    /// False also means that no graph boundary measurements were collected,
    /// for example on the specialized pipeline path.
    pub fn has_complete_boundary_observations(&self) -> bool {
        self.boundary_observations > 0 && self.incomplete_observations == 0
    }
}
