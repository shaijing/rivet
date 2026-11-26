//! Physical lowering and execution graph for compiled image pipelines.

mod compile;
mod physical;

pub(super) use compile::{register_execution_kernels, select_terminal_normalize_layout};
pub(crate) use physical::ImageExecutionGraph;
pub use physical::ImageGraphInfo;
