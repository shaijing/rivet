//! Physical lowering and execution infrastructure for Rivet.
//!
//! Backend runtime resources belong here or in `rivet-core`; they must never
//! be stored in the backend-independent logical plan.

pub mod cache;
pub mod runtime;
