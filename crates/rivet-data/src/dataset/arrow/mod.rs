//! Arrow IPC backend types.
//!
//! These types are intentionally kept under the Arrow namespace. They are
//! storage adapters, not modality-neutral dataset abstractions.

mod mmap;
mod row;
mod table;

pub use row::ArrowRow;
pub use table::MmapArrowTable;
