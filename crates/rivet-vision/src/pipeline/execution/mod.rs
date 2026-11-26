//! Physical lowering and execution graph for compiled image pipelines.

mod compile;
mod physical;

pub(super) use compile::expand_terminal_sample_fusion;
pub(crate) use physical::ImageExecutionGraph;
pub use physical::ImageGraphInfo;
