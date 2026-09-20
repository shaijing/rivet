//! Deterministic semantic random namespaces for data pipelines.
//!
//! Random values are derived from semantic identities rather than execution
//! order. The algorithm and constants in this module are part of the V1
//! reproducibility contract.

mod context;
mod key;
mod mix;
mod stream;

pub use context::{DOMAIN_SAMPLER, DOMAIN_TENSOR, DOMAIN_TRANSFORM, RandomContext, RandomDomain};
pub use key::{OpKey, SampleKey};
pub use mix::{RNG_ALGORITHM_VERSION, combine, mix_u64};
pub use stream::RandomStream;
