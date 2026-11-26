//! Image pipeline construction, planning, and execution.

mod builder;
mod execution;
pub mod op;
mod planning;
mod transform;

pub use builder::ImagePipeline;
pub(crate) use execution::ImageExecutionGraph;
pub use execution::ImageGraphInfo;
pub use planning::{ImageConcat, ImageOptimizationOptions};
pub use transform::{Compose, ImageTransform, TransformSequence};

#[cfg(test)]
mod tests;
