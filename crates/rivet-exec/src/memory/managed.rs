//! Shared byte reservations for executor-owned work.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug)]
pub struct ManagedMemoryBudget(Arc<Budget>);

#[derive(Debug)]
struct Budget {
    limit: usize,
    held: AtomicUsize,
    peak: AtomicUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManagedMemorySnapshot {
    pub limit_bytes: usize,
    pub reserved_bytes: usize,
    pub peak_reserved_bytes: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("morsel requires {requested} managed bytes, above the executor budget of {limit}")]
pub struct MorselExceedsBudget {
    pub requested: usize,
    pub limit: usize,
}

impl ManagedMemoryBudget {
    pub fn new(limit_bytes: usize) -> Self {
        Self(Arc::new(Budget {
            limit: limit_bytes,
            held: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }))
    }

    /// Returns None when another admitted morsel holds too much capacity.
    /// The caller can drain completed work before retrying.
    pub fn try_reserve(
        &self,
        bytes: usize,
    ) -> Result<Option<MemoryReservation>, MorselExceedsBudget> {
        if bytes > self.0.limit {
            return Err(MorselExceedsBudget {
                requested: bytes,
                limit: self.0.limit,
            });
        }
        let limit_before = self.0.limit - bytes;
        let mut previous = self.0.held.load(Ordering::Acquire);
        loop {
            if previous > limit_before {
                return Ok(None);
            }
            match self.0.held.compare_exchange_weak(
                previous,
                previous + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => previous = actual,
            }
        }
        self.0.peak.fetch_max(previous + bytes, Ordering::Relaxed);
        Ok(Some(MemoryReservation {
            budget: self.clone(),
            bytes,
        }))
    }

    pub fn snapshot(&self) -> ManagedMemorySnapshot {
        let reserved_bytes = self.0.held.load(Ordering::Acquire);
        ManagedMemorySnapshot {
            limit_bytes: self.0.limit,
            reserved_bytes,
            peak_reserved_bytes: self.0.peak.load(Ordering::Relaxed).max(reserved_bytes),
        }
    }
}

/// Held capacity is released by the owner or when its last Arc handle drops.
#[derive(Debug)]
pub struct MemoryReservation {
    budget: ManagedMemoryBudget,
    bytes: usize,
}

impl MemoryReservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Return unused allowance while maintaining ownership of the retained bytes.
    pub fn shrink_to(&mut self, bytes: usize) {
        assert!(bytes <= self.bytes, "reservation cannot grow by shrinking");
        self.budget
            .0
            .held
            .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        self.bytes = bytes;
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.budget.0.held.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Observes addressable storage only while a tensor or view owns it. Weak
/// handles prevent reuse of the storage identity without keeping data alive.
#[derive(Default)]
pub(crate) struct ManagedBackings {
    allocations: std::collections::HashMap<usize, (std::sync::Weak<rivet_core::Storage>, usize)>,
}

impl ManagedBackings {
    pub(crate) fn insert(&mut self, tensor: &rivet_core::Tensor) -> usize {
        let bytes = tensor.storage_bytes();
        if bytes == 0 {
            return 0;
        }
        let backing = tensor.weak_storage();
        let id = backing.as_ptr() as usize;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.allocations.entry(id) {
            slot.insert((backing, bytes));
            bytes
        } else {
            0
        }
    }

    pub(crate) fn live_bytes(&mut self) -> usize {
        self.allocations
            .retain(|_, (backing, _)| backing.strong_count() != 0);
        self.allocations.values().map(|(_, bytes)| *bytes).sum()
    }
}
