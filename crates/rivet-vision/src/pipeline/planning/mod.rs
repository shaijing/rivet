//! Logical plan construction, property inference, and optimization.

mod cached_inference;
mod canonicalize;
mod cse;
mod dag;
pub(super) mod inference;
pub(super) mod logical;
pub(super) mod optimizer;
mod options;
mod semantics;

pub use dag::ImageConcat;
pub use options::ImageOptimizationOptions;
