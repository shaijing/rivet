use crate::storage::Storage;
use crate::{DType, Device, Layout};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

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
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    dtype: DType,
    device: Device,
}

/// A cheap, reference-counted tensor handle.
#[derive(Debug, Clone)]
pub struct Tensor(Arc<Tensor_>);
