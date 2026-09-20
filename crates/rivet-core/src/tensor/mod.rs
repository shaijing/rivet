use crate::storage::Storage;
use crate::{DType, Device, Layout};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod access;
mod construction;
mod convert;
mod index;
mod ops;
mod view;

pub use construction::RangeElement;

/// Unique identifier for a logical tensor node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    fn new() -> Self {
        static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
struct Tensor_ {
    id: TensorId,
    /// Immutable backing allocation shared by view tensors.
    storage: Arc<Storage>,
    layout: Layout,
    dtype: DType,
    device: Device,
}

/// A cheap, reference-counted tensor handle.
#[derive(Debug, Clone)]
pub struct Tensor(Arc<Tensor_>);

/// A tensor whose handle and backing storage have been moved out of the
/// reference-counted [`Tensor`] representation.
///
/// This type is intentionally not `Clone`: it is the proof object used by
/// mutable zero-copy consumers such as an ownership-transferring DLPack
/// export. Dropping it releases the backing allocation.
#[derive(Debug)]
pub struct ExclusiveTensor {
    pub(crate) storage: crate::storage::Storage,
    pub(crate) layout: Layout,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

impl ExclusiveTensor {
    /// Consumes the exclusive tensor and returns its owned storage metadata.
    pub fn into_parts(self) -> (crate::storage::Storage, Layout, DType, Device) {
        let Self {
            storage,
            layout,
            dtype,
            device,
        } = self;
        (storage, layout, dtype, device)
    }
}
